"""Движение кадра по кадрам — чистая математика петли, без torch / cv2 / runpod.

Почему отдельный модуль: handler.py на импорте тянет torch, cv2 и регистрирует
serverless-хендлер, а проверить математику движения надо там, где нет ни
видеокарты, ни 15 ГБ весов (motion_selftest.py, стенд без SD). Всё, что решает,
КАК двигается камера в кадре i, живёт здесь; handler только исполняет.

Две вещи (контракт — README, раздел 2026-10-24):

  · «Пульс под музыку» (`react`, `energy`, `hits`): зум, поворот и сдвиг кадра
    следуют громкости песни, а на сильных долях — короткий «толчок» зума и
    силы перерисовки. Это главная фича Neural Frames, и warp-петле она
    достаётся почти даром: камера и так двигается каждый кадр, мы лишь решаем,
    НАСКОЛЬКО.
  · Промежуточный кадр для 24 к/с по ИЗВЕСТНОМУ движению (`half_affine`):
    мы сами знаем, какой аффинкой кадр i+1 получен из кадра i, поэтому
    середину шага строим точно, а не угадываем оптическим потоком.

⚠️⚠️ Выключено (react = 0 или нет поля) — массивы РОВНО те константы, что петля
брала до 24.10: тот же объект float из STYLES, тот же `steps_for`. Старый вызов
обязан давать побайтно тот же ролик; motion_selftest.py это сторожит.
"""
import math

# --- «пульс»: числа и почему такие -----------------------------------------
# Множитель движения от громкости e∈[0,1]: 0.35 + 1.3·e. Тишина не замирает
# совсем (0.35 — камера дышит и в паузе), полный удар даёт 1.65 до нормировки.
ENERGY_FLOOR = 0.35
ENERGY_GAIN = 1.3
# Пределы нормированного множителя одного кадра. Без нижнего тихое вступление
# стоит колом, без верхнего один громкий кадр в тихой песне после нормировки
# получал бы ×4 зума за 1/12 с — рывок, а не пульс.
MULT_MIN = 0.25
MULT_MAX = 2.2
# ⚠️⚠️ Потолок СРЕДНЕГО множителя за шот. Снос композиции копится по шоту
# (README, «три вещи», п. 1: зум 0.6 %/кадр удваивает кадр за 9 с), и громкий
# припев не имеет права увести камеру «внутрь глаза» быстрее, чем стиль сегодня.
# 1.3 — шот может двигаться на треть живее среднего, не больше; средний по
# клипу держится на 1.0 (см. _balance).
SHOT_CAP = 1.3
# Толчок на сильной доле: доп. зум и доп. сила перерисовки с затуханием.
# 0.010 зума — это половина шага стиля flow за один кадр: глаз видит «удар»,
# а композиция за 4 кадра почти не сдвигается. +0.08 силы — img2img на кадре
# удара перерисовывает заметно больше, и картинка «вздрагивает».
PUNCH_ZOOM = 0.010
PUNCH_STRENGTH = 0.08
PUNCH_DECAY = 0.55          # на кадр при 12 к/с: 1 → .55 → .30 → .17
PUNCH_SECONDS = 4 / 12      # ~4 кадра при 12 к/с; при другой fps — те же 0.33 с
# Безопасная полоса силы для петли. Ниже 0.30 — постеризация (README, п. 3),
# выше 0.62 кадр перестаёт наследовать предыдущий и петля превращается в
# слайд-шоу. Верх на 0.02 выше потолка ручного `strength` (0.60): толчку нужен
# хоть какой-то ход и у стиля с переопределённой силой.
STRENGTH_BAND = (0.30, 0.62)

MAX_HITS = 2000
# `energy` может быть на секунду длиннее клипа: браузер считает огибающую по
# окну песни и округляет иначе, чем мы `seconds × fps`. Больше — это уже не
# округление, а чужой массив, и молча его резать значит рисовать не ту песню.
ENERGY_SLACK_FRAMES = 12


def steps_for(strength, eff_steps):
    """Держим ЭФФЕКТИВНЫЕ шаги: img2img запускает только долю strength от них.
    (Та же формула, что `_steps_for` в handler.py — он зовёт эту.)"""
    return max(int(round(eff_steps / max(strength, 0.05))), 4)


# ------------------------------------------------------------- разбор входа
def _finite(x):
    # bool — подкласс int; True в массиве громкости — ошибка вызывающего, а не 1.0.
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError("not a number")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("not finite")
    return x


def parse_react(raw):
    """None/0 → 0.0 (выключено). Вне [0, 1] или не число → ValueError."""
    if raw is None:
        return 0.0
    r = _finite(raw)
    if not (0.0 <= r <= 1.0):
        raise ValueError("react out of range")
    return r


def parse_energy(raw, total):
    """Огибающая громкости по кадрам генерации → список длины total.

    Короче клипа — держим последнее значение (хвост песни после последнего окна
    огибающей). Длиннее total + ENERGY_SLACK_FRAMES или не числа → ValueError.
    Значения зажимаются в [0, 1]: браузер уже нормировал, а лишняя сотая от
    округления не повод отказывать в оплаченном клипе.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("energy is not a list")
    if len(raw) > total + ENERGY_SLACK_FRAMES:
        raise ValueError("energy longer than the clip")
    vals = [min(1.0, max(0.0, _finite(v))) for v in raw]
    if not vals:
        return None
    if len(vals) < total:
        vals += [vals[-1]] * (total - len(vals))
    return vals[:total]


def parse_hits(raw, total):
    """Индексы кадров сильных долей → отсортированный список без повторов.

    Больше MAX_HITS или не числа → ValueError; вне [0, total) — молча мимо:
    браузер ставит доли по окну песни, и доля на последней миллисекунде
    окна — не ошибка, а округление.
    """
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("hits is not a list")
    if len(raw) > MAX_HITS:
        raise ValueError("too many hits")
    out = set()
    for v in raw:
        i = int(round(_finite(v)))
        if 0 <= i < total:
            out.add(i)
    return sorted(out)


# ------------------------------------------------------------------ кривые
def _shot_ids(total, shot_starts):
    """Номер шота для каждого кадра (шот = от начала до следующего начала)."""
    ids, cur = [], -1
    for i in range(total):
        if i in shot_starts or cur < 0:
            cur += 1
        ids.append(cur)
    return ids


def _balance(vals, frames, groups, cap, iters=12):
    """Отнормировать `vals` на кадрах `frames`: среднее по клипу = 1, среднее
    по каждому шоту ≤ cap. Возвращает новый список (вне `frames` — как было).

    Громкий шот срезается до потолка, а недостача отдаётся тихим шотам — так
    общий бюджет движения по клипу остаётся ровно тем, что у стиля сегодня
    (иначе потолок молча делал бы клип статичнее обещанного стиля).
    """
    v = list(vals)
    if not frames:
        return v
    n = float(len(frames))
    by_group = {}
    for i in frames:
        by_group.setdefault(groups[i], []).append(i)

    s = sum(v[i] for i in frames)
    if s <= 1e-12:
        return v
    k = n / s
    for i in frames:
        v[i] *= k

    for _ in range(iters):
        capped_sum, free = 0.0, []
        for g, idx in by_group.items():
            gm = sum(v[i] for i in idx) / len(idx)
            if gm > cap + 1e-9:
                f = cap / gm
                for i in idx:
                    v[i] *= f
            if sum(v[i] for i in idx) / len(idx) >= cap - 1e-9:
                capped_sum += sum(v[i] for i in idx)
            else:
                free.append(idx)
        free_sum = sum(v[i] for idx in free for i in idx)
        if free_sum <= 1e-12:
            break                       # все шоты в потолке — больше не поднять
        f = (n - capped_sum) / free_sum
        if abs(f - 1.0) < 1e-9:
            break
        for idx in free:
            for i in idx:
                v[i] *= f
    return v


def _smooth(vals, fps):
    """Центрированное треугольное сглаживание огибающей (~0.4 с, без задержки).

    Громкость — МЕДЛЕННАЯ составляющая (куплет/припев, фраза), удар — быстрая,
    и её несут `hits`. Без сглаживания покадровые скачки огибающей дают рывки
    скорости зума (камера «дрожит»), а скачок громкости на кадр позже доли
    перебивал спад толчка — удар переставал быть пиком (поймано самопроверкой
    на огибающей с долями не в фазе с хитами).
    """
    r = max(1, int(round(fps / 6.0)))            # 12 к/с → 2 → окно 5 кадров
    w = [r + 1 - abs(j) for j in range(-r, r + 1)]
    n = len(vals)
    out = []
    for i in range(n):
        acc = ws = 0.0
        for j, wj in zip(range(-r, r + 1), w):
            k = i + j
            if 0 <= k < n:
                acc += vals[k] * wj
                ws += wj
        out.append(acc / ws)
    return out


def _punch_envelope(total, hits, fps):
    """Огибающая толчков: мгновенная атака на доле, экспоненциальный спад.

    ⚠️ Перекрытие — максимум, а не сумма: при долях каждый кадр (2000 хитов —
    допустимый вход) сумма давала бы двойной зум на весь клип, а максимум
    ограничен одним толчком, что бы ни прислали.
    """
    env = [0.0] * total
    if not hits:
        return env
    decay = PUNCH_DECAY ** (12.0 / fps)
    width = max(2, int(math.ceil(PUNCH_SECONDS * fps)))
    for k in hits:
        for j in range(width):
            i = k + j
            if i >= total:
                break
            val = decay ** j
            if val > env[i]:
                env[i] = val
    return env


def motion_curves(total, fps, style, react=0.0, energy=None, hits=None,
                  shot_starts=(0,), eff_steps=3.5):
    """Параметры перехода (i-1 → i) для каждого кадра i.

    Возвращает dict списков длины total: zoom, angle, tx, ty, strength, steps.
    Значения на кадрах начала шота не используются (кадр рисуется заново), но
    заполнены — чтобы индекс всегда был номером кадра.

    react = 0 → константы стиля, те же объекты float. Иначе:
      mult_i = mix(1, норм(0.35 + 1.3·e_i), react)   — громкость
      zoom_i = 1 + (zoom − 1)·mult_i + react·0.010·env_i   — плюс толчок,
               и вся зум-дельта ещё раз нормирована (_balance): толчки и
               громкость перераспределяют, КОГДА камера едет, но не сколько;
      angle/tx/ty = стиль × mult_i;
      strength_i = clamp(strength + react·0.08·env_i, 0.30, 0.62),
      steps_i = steps_for(strength_i).
    """
    z0, a0 = style["zoom"], style["angle"]
    tx0, ty0, s0 = style["tx"], style["ty"], style["strength"]
    react = float(react or 0.0)
    if react <= 0.0:
        st = steps_for(s0, eff_steps)
        return {"zoom": [z0] * total, "angle": [a0] * total,
                "tx": [tx0] * total, "ty": [ty0] * total,
                "strength": [s0] * total, "steps": [st] * total}

    starts = set(shot_starts)
    loop = [i for i in range(total) if i not in starts]
    groups = _shot_ids(total, starts)
    e = _smooth(energy, fps) if energy is not None else [0.5] * total

    # Громкость → множитель, нормированный к среднему 1 по кадрам петли, с
    # пределами на кадр. Несколько проходов: зажим сдвигает среднее.
    raw = [ENERGY_FLOOR + ENERGY_GAIN * x for x in e]
    nrm = list(raw)
    if loop:
        for _ in range(6):
            m = sum(nrm[i] for i in loop) / len(loop)
            nrm = [min(MULT_MAX, max(MULT_MIN, x / m)) for x in nrm]
    mult = [1.0 + react * (x - 1.0) for x in nrm]
    mult = _balance(mult, loop, groups, SHOT_CAP)

    env = _punch_envelope(total, hits or [], fps)

    base = z0 - 1.0
    if base > 1e-9:
        # Дельта зума в долях дельты стиля: громкость + толчок, и одна общая
        # нормировка — бюджет зума по клипу тот же, что сегодня.
        rel = [mult[i] + react * PUNCH_ZOOM * env[i] / base for i in range(total)]
        rel = _balance(rel, loop, groups, SHOT_CAP)
        zoom = [1.0 + base * x for x in rel]
    else:
        # Стиль без зума (такого сейчас нет): масштабировать нечего, толчок —
        # чистая добавка, потому что бюджета, который он мог бы съесть, нет.
        zoom = [z0 + react * PUNCH_ZOOM * env[i] for i in range(total)]

    lo, hi = STRENGTH_BAND
    strength = [min(hi, max(lo, s0 + react * PUNCH_STRENGTH * env[i]))
                for i in range(total)]
    return {
        "zoom": zoom,
        "angle": [a0 * m for m in mult],
        "tx": [tx0 * m for m in mult],
        "ty": [ty0 * m for m in mult],
        "strength": strength,
        "steps": [steps_for(s, eff_steps) for s in strength],
    }


# ------------------------------------------------- середина известного шага
def rotation_matrix(cx, cy, angle_deg, scale):
    """То же, что cv2.getRotationMatrix2D, — без cv2 (для самопроверки)."""
    t = math.radians(angle_deg)
    a, b = scale * math.cos(t), scale * math.sin(t)
    return [[a, b, (1 - a) * cx - b * cy],
            [-b, a, b * cx + (1 - a) * cy]]


def step_matrix(w, h, zoom, angle, tx, ty):
    """Аффинка шага петли — ровно та, что строит `_warp` в handler.py."""
    m = rotation_matrix(w / 2, h / 2, angle, zoom)
    m[0][2] += tx
    m[1][2] += ty
    return m


def half_affine(w, h, zoom, angle, tx, ty):
    """A такая, что A∘A = шаг петли M (точный квадратный корень, не «половина
    чисел»). Линейная часть подобия: sqrt(zoom) и angle/2 — у подобий это
    точный корень; сдвиг b из A∘A = M: (H + I)·b = m.

    Зачем корень, а не «zoom/2»: при zoom 1.022 и сдвиге с центром разница
    доли пикселя на шаг, но она копится в дрожь середины относительно соседей.
    """
    t = math.radians(angle) / 2.0
    s = math.sqrt(zoom)
    a, b = s * math.cos(t), s * math.sin(t)
    m = step_matrix(w, h, zoom, angle, tx, ty)
    mx, my = m[0][2], m[1][2]
    # (H + I) = [[a+1, b], [-b, a+1]] — подобие, обратная выписывается сразу.
    p, q = a + 1.0, b
    det = p * p + q * q
    bx = (p * mx - q * my) / det
    by = (q * mx + p * my) / det
    return [[a, b, bx], [-b, a, by]]
