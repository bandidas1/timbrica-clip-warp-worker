"""Самопроверка motion.py — без torch, без видеокарты, без весов.

Зачем: «пульс под музыку» меняет то, как петля двигает камеру, а проверить
это прогоном SD на стенде больше нечем (веса со стенда сняты, 15 ГБ ради
арифметики не качаем). Математика движения вынесена в motion.py как раз
для того, чтобы её можно было прогнать здесь за секунду.

Что держит:
  · выключено → РОВНО старые константы (те же объекты float) — иначе старый
    вызов перестал бы давать побайтно тот же ролик;
  · включено → средний множитель зума по клипу в ±15 % от стиля, по шоту не
    выше потолка, — композиция не уползает быстрее, чем сегодня;
  · доли дают локальные максимумы зума и силы — «удар» виден;
  · сила всегда в безопасной полосе, а число вызовов UNet на кадр (формула
    diffusers 0.40 `int(steps × strength)`) не меняется — толчки не дороже;
  · длины energy: короче — держим последнее, длиннее клипа + 12 — отказ;
  · середина шага для 24 к/с: A∘A = шаг петли и совпадает с cv2 (если есть).

Запуск: python motion_selftest.py   (код 0 — всё сошлось)
"""
import math
import random
import sys

import motion as M

# Копия STYLES из handler.py — импортировать handler нельзя (torch). Сверка,
# что копия не разошлась с оригиналом, — текстом, ниже в _styles_match().
STYLES = {
    "flow":  {"zoom": 1.018, "angle": 0.35, "tx": 0.0,  "ty": 0.0, "strength": 0.52},
    "drift": {"zoom": 1.012, "angle": 0.00, "tx": -1.5, "ty": 0.0, "strength": 0.44},
    "push":  {"zoom": 1.022, "angle": 0.12, "tx": 0.0,  "ty": 0.0, "strength": 0.48},
    "orbit": {"zoom": 1.008, "angle": 0.55, "tx": 0.8,  "ty": 0.0, "strength": 0.46},
}
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def _styles_match():
    import os
    import re
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "handler.py"),
               encoding="utf-8").read()
    for k, v in STYLES.items():
        line = re.search(r'"%s":\s*\{([^}]*)\}' % k, src)
        check(line is not None, "стиль %s не найден в handler.py" % k)
        if not line:
            continue
        got = dict((a, float(b)) for a, b in re.findall(r'"(\w+)":\s*(-?[\d.]+)', line.group(1)))
        check(got == v, "копия стиля %s разошлась с handler.py: %r != %r" % (k, got, v))


def unet_evals(steps, strength):
    """diffusers 0.40, StableDiffusionImg2ImgPipeline.get_timesteps."""
    init = min(int(steps * strength), steps)
    return steps - max(steps - init, 0)


def song(total, fps=12, bpm=124, seed=1):
    """Синтетическая огибающая песни: куплет/припев + удары на долях."""
    rnd = random.Random(seed)
    beat = fps * 60.0 / bpm
    hits = [int(round(k * beat)) for k in range(int(total / beat) + 1) if round(k * beat) < total]
    e = []
    for i in range(total):
        section = 0.35 if (i // (16 * fps)) % 2 == 0 else 0.75   # куплет / припев
        ph = (i % beat) / beat
        pulse = math.exp(-ph * 5.0)                              # удар и спад
        e.append(min(1.0, max(0.0, section + 0.25 * pulse + rnd.uniform(-0.03, 0.03))))
    return e, hits


def starts_every(total, n):
    return set(range(0, total, n))


def test_off_is_identical():
    for name, st in STYLES.items():
        for kw in ({}, {"react": 0.0}, {"react": 0, "energy": [0.9] * 50, "hits": [3, 9]}):
            c = M.motion_curves(120, 12, st, shot_starts=starts_every(120, 96), **kw)
            for key in ("zoom", "angle", "tx", "ty", "strength"):
                check(len(c[key]) == 120, "off %s: длина %s" % (name, key))
                check(all(x is st[key] for x in c[key]),
                      "off %s: %s не тот же объект, что в STYLES" % (name, key))
            old_steps = max(int(round(3.5 / max(st["strength"], 0.05))), 4)
            check(c["steps"] == [old_steps] * 120, "off %s: шаги разошлись со старой формулой" % name)


def test_on_keeps_the_budget():
    total = 12 * 60
    starts = starts_every(total, 96)
    e, hits = song(total)
    for name, st in STYLES.items():
        for react in (0.3, 0.7, 1.0):
            c = M.motion_curves(total, 12, st, react, e, hits, starts)
            loop = [i for i in range(total) if i not in starts]
            zm = sum((c["zoom"][i] - 1) / (st["zoom"] - 1) for i in loop) / len(loop)
            check(abs(zm - 1) <= 0.15, "%s r=%.1f: средний зум ×%.3f вне ±15 %%" % (name, react, zm))
            if st["angle"]:
                am = sum(c["angle"][i] / st["angle"] for i in loop) / len(loop)
                check(abs(am - 1) <= 0.15, "%s r=%.1f: средний поворот ×%.3f" % (name, react, am))
            # по шотам — не выше потолка (накопление зума внутри шота)
            for s0 in sorted(starts):
                idx = [i for i in range(s0 + 1, min(s0 + 96, total))]
                if not idx:
                    continue
                sm = sum((c["zoom"][i] - 1) / (st["zoom"] - 1) for i in idx) / len(idx)
                check(sm <= M.SHOT_CAP + 1e-6, "%s r=%.1f шот %d: ×%.3f выше потолка" % (name, react, s0, sm))
                cum_new = math.prod(c["zoom"][i] for i in idx)
                cum_old = st["zoom"] ** len(idx)
                check(math.log(cum_new) <= M.SHOT_CAP * math.log(cum_old) + 1e-6,
                      "%s шот %d: накопленный зум %.2f против %.2f" % (name, s0, cum_new, cum_old))
            # множитель реально следует громкости: припев живее куплета
            verse = [i for i in loop if (i // 192) % 2 == 0]
            chorus = [i for i in loop if (i // 192) % 2 == 1]
            mv = sum(c["zoom"][i] for i in verse) / len(verse)
            mc = sum(c["zoom"][i] for i in chorus) / len(chorus)
            check(mc > mv, "%s r=%.1f: припев (%.4f) не живее куплета (%.4f)" % (name, react, mc, mv))


def test_hits_are_peaks():
    total = 12 * 30
    starts = starts_every(total, 96)
    for name, st in STYLES.items():
        for energy in (None, [0.6] * total, song(total, seed=3)[0]):
            hits = list(range(10, total - 5, 9))          # доли реже толчка
            c = M.motion_curves(total, 12, st, 1.0, energy, hits, starts)
            for k in hits:
                if k in starts or k - 1 in starts or k + 1 in starts:
                    continue
                z = c["zoom"]
                check(z[k] > z[k - 1] and z[k] > z[k + 1],
                      "%s: доля %d не пик зума (%.5f %.5f %.5f)" % (name, k, z[k - 1], z[k], z[k + 1]))
                s = c["strength"]
                check(s[k] > s[k - 1] and s[k] >= s[k + 1], "%s: доля %d не пик силы" % (name, k))


def test_strength_band_and_gpu_cost():
    total = 12 * 20
    for name, st in STYLES.items():
        for s0 in (st["strength"], 0.30, 0.60):
            st2 = dict(st, strength=s0)
            c = M.motion_curves(total, 12, st2, 1.0, None, list(range(0, total, 2)), {0})
            lo, hi = M.STRENGTH_BAND
            check(all(lo <= s <= hi for s in c["strength"]), "%s s0=%.2f: сила вне полосы" % (name, s0))
            for s, n in zip(c["strength"], c["steps"]):
                check(n == M.steps_for(s, 3.5), "шаги не из steps_for")
                check(abs(n * s - 3.5) <= s / 2 + 1e-9, "эффективные шаги %.2f далеко от 3.5" % (n * s))
                check(unet_evals(n, s) == unet_evals(M.steps_for(st["strength"], 3.5), st["strength"]),
                      "%s: вызовов UNet на кадре удара больше, чем без него (s=%.2f)" % (name, s))
    # и по всей полосе — чтобы потолок не сдвинул цену прогона незаметно
    evals = {unet_evals(M.steps_for(s / 100, 3.5), s / 100) for s in range(30, 63)}
    check(evals == {3}, "вызовов UNet по полосе силы: %r, ждали {3}" % evals)


def test_inputs():
    total = 100
    check(M.parse_energy([0.2, 0.4], total) == [0.2] + [0.4] * 99, "короткая energy не добита последним")
    check(len(M.parse_energy([0.5] * (total + 12), total)) == total, "energy +12 не принята/не обрезана")
    for bad in ([0.5] * (total + 13), ["x"], [float("nan")], [True], "0.5", 5):
        try:
            M.parse_energy(bad, total)
            check(False, "energy %r принята" % (bad if not isinstance(bad, list) else bad[:2]))
        except ValueError:
            pass
    check(M.parse_energy([-1, 2], 2) == [0.0, 1.0], "energy не зажата в [0,1]")
    check(M.parse_energy([], total) is None, "пустая energy не равна отсутствию")
    check(M.parse_hits([5, 5, -1, 100, 99, 3.0], total) == [3, 5, 99], "хиты: вне диапазона не отброшены")
    for bad in ([0] * 2001, ["a"], [None], "1,2"):
        try:
            M.parse_hits(bad, total)
            check(False, "hits %r принят" % (bad[:2] if isinstance(bad, list) else bad))
        except ValueError:
            pass
    check(M.parse_react(None) == 0.0 and M.parse_react(0.4) == 0.4, "react разбор")
    for bad in (-0.1, 1.5, "1", float("inf")):
        try:
            M.parse_react(bad)
            check(False, "react %r принят" % bad)
        except ValueError:
            pass
    # все длины кривых = total, в т.ч. крошечный клип
    for t in (2, 3, 30):
        c = M.motion_curves(t, 12, STYLES["flow"], 1.0, M.parse_energy([1.0], t), [0, 1], {0})
        check(all(len(v) == t for v in c.values()), "длины кривых при total=%d" % t)


def test_half_affine():
    def compose(a, b):        # a∘b: сначала b, потом a
        return [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1],
                 a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1],
                 a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2]]]
    for (w, h) in ((432, 768), (768, 432), (512, 512)):
        for st in STYLES.values():
            for k in (1.0, 2.2, 0.25):
                z = 1 + (st["zoom"] - 1) * k
                args = (w, h, z, st["angle"] * k, st["tx"] * k, st["ty"] * k)
                a = M.half_affine(*args)
                m = M.step_matrix(*args)
                aa = compose(a, a)
                err = max(abs(aa[r][c] - m[r][c]) for r in range(2) for c in range(3))
                check(err < 1e-9, "A∘A ≠ M (%d×%d, err %.2e)" % (w, h, err))
    try:
        import cv2
        m = M.step_matrix(432, 768, 1.018, 0.35, -1.5, 0.7)
        ref = cv2.getRotationMatrix2D((216.0, 384.0), 0.35, 1.018)
        ref[0, 2] += -1.5
        ref[1, 2] += 0.7
        err = max(abs(ref[r, c] - m[r][c]) for r in range(2) for c in range(3))
        check(err < 1e-9, "step_matrix ≠ cv2.getRotationMatrix2D (err %.2e)" % err)
    except ImportError:
        print("  (cv2 нет — сверка с getRotationMatrix2D пропущена)")


def main():
    for t in (_styles_match, test_off_is_identical, test_on_keeps_the_budget,
              test_hits_are_peaks, test_strength_band_and_gpu_cost, test_inputs,
              test_half_affine):
        n = len(FAILS)
        t()
        print("%s %s" % ("ok  " if len(FAILS) == n else "FAIL", t.__name__))
    for f in FAILS[:40]:
        print("  ·", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
