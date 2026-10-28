"""RunPod serverless handler — видеоклип по промпту через warp-петлю SD 1.5 + LCM.

Контракт и обоснования — README.md рядом. Три вещи, которые здесь выглядят как
мелочи, а на самом деле держат результат (все три из замера 2026-09-03):
  · склейки каждые shot_seconds — иначе композиция уползает и клип невозможен;
  · коррекция цвета к первому кадру ШОТА — иначе палитра уходит в сепию;
  · эффективные шаги = steps × strength — иначе картинка постеризуется.

Математика движения (пульс под музыку, середина шага для 24 к/с) — в motion.py:
её проверяет motion_selftest.py там, где нет ни видеокарты, ни весов.
"""
import base64
import gc
import io
import hashlib
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np
import runpod
import torch
from PIL import Image

import motion

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/sd15")
LORA_DIR = os.environ.get("LORA_DIR", "/models/lcm-lora-sdv1-5")
# Пусто = обычные имена файлов (так лежит кэш HF при локальном прогоне),
# "fp16" = то, что запечено в образ. Значение по умолчанию берётся от образа,
# а local_smoke.py его перекрывает пустой строкой.
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "fp16") or None

NEG = ("watermark, text, signature, blurry, low quality, jpeg artifacts, "
       "deformed, extra limbs")

# Наборы параметров движения. Значения замерены на живых прогонах: зум держится
# СИЛЬНО ниже интуитивного — накопление за сотню кадров быстрее, чем кажется.
STYLES = {
    "flow":  {"zoom": 1.018, "angle": 0.35, "tx": 0.0,  "ty": 0.0, "strength": 0.52},
    "drift": {"zoom": 1.012, "angle": 0.00, "tx": -1.5, "ty": 0.0, "strength": 0.44},
    "push":  {"zoom": 1.022, "angle": 0.12, "tx": 0.0,  "ty": 0.0, "strength": 0.48},
    "orbit": {"zoom": 1.008, "angle": 0.55, "tx": 0.8,  "ty": 0.0, "strength": 0.46},
}

# Потолок длины клипа. ⚠️ Раньше здесь стояло 30 секунд — это был предел
# ТРАНСПОРТА (инлайн base64 не тащит больше), а не движка. С появлением отдачи
# файла приложению (см. `_deliver`) ограничение сняли до 5 минут — верхнего
# тарифного тира. Выше не надо: 5 минут это уже ~13 минут живого GPU.
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", 300))

# Разрешённые размеры кадра — зеркало config/ai-video.php → tools.clip-generator
# .aspects (w × h). Квадрат, 9:16 и 16:9. 640×640 остаётся из прежнего контракта.
# ⚠️⚠️ Пробы ОБЯЗАНЫ быть в этом списке. До 04.09 его не было вовсе, а проверка
# гласила `size not in (512, 640)` — и проба, которую приложение шлёт размером
# 384, отвергалась с `bad_params` КАЖДЫЙ раз. Функция, которой мы бьём главную
# жалобу рынка («платишь до того, как увидишь»), не могла отработать ни разу.
ALLOWED_SIZES = {
    (512, 512), (640, 640),          # квадрат (640 — из прежнего контракта)
    (432, 768), (768, 432),          # 9:16 и 16:9
    (384, 384), (288, 512), (512, 288),   # пробы тех же трёх форм
}

# Потолок инлайн-выдачи. Ниже настоящего предела RunPod с запасом: base64
# раздувает байты на треть, и упереться в предел ПОСЛЕ прогона — значит выбросить
# уже потраченные GPU-минуты.
INLINE_MAX_BYTES = int(os.environ.get("INLINE_MAX_BYTES", 6 * 1024 * 1024))
EFF_STEPS = float(os.environ.get("EFF_STEPS", 3.5))

# Выход от этой площади (1080×1080 и больше) кодируется с потолком битрейта —
# см. _Encoder. 1280×720 (921 600) ниже порога намеренно: прежние размеры
# выхода кодируются ровно той командой, что и до 24.10.
BIG_OUT_AREA = 1080 * 1080
BIG_OUT_MAXRATE = "10M"
BIG_OUT_BUFSIZE = "20M"
# Очередь кадров к кодировщику (см. _Encoder): ~4 с при 12 к/с, ~1 МБ на кадр.
QUEUE_FRAMES = int(os.environ.get("QUEUE_FRAMES", 48))

_PIPES = {"t2i": None, "i2i": None}


def _cpu_budget():
    """Сколько ядер контейнеру ДАЛИ, а не сколько он ВИДИТ.

    ⚠️⚠️ Контейнер RunPod видит все ядра хоста, а выделено ему 16 vCPU
    (`/v2/pods/{id}` → gpu.vcpuCount, 4090, 24.10). x264 с `threads=auto`
    (1.5 × видимых) и OpenCV (по потоку на видимое ядро) заводят потоки на
    чужие ядра, квота cgroup их душит — и вместе с ними поток, который кормит
    GPU. Отсюда разброс живого A/B: тот же прогон +7 % и +37 %.
    """
    seen = os.cpu_count() or 1
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = seen
    quota = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:          # cgroup v2
            q, p = f.read().split()[:2]
            if q != "max":
                quota = int(q) / int(p)
    except (OSError, ValueError):
        try:                                                # cgroup v1
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                q = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                p = int(f.read())
            if q > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    budget = aff if quota is None else max(1, min(aff, int(math.ceil(quota))))
    return budget, {"visible": seen, "affinity": aff,
                    "quota": round(quota, 2) if quota else None, "budget": budget}


CPU_BUDGET, CPU_INFO = _cpu_budget()
# OpenCV: не больше 4 потоков. Кадр 432×768 — миллисекунды и в один поток, а
# сотня потоков на чужие ядра — это троттлинг. На байты не влияет (проверено
# 24.10: warpAffine LANCZOS/LINEAR, LAB, addWeighted при 1/2/4/8/32 потоках
# дают один и тот же результат).
cv2.setNumThreads(max(1, min(4, CPU_BUDGET)))
# x264 у новых выходов: бюджет минус два ядра петле (Python + драйвер CUDA),
# не больше 8 — для 12 к/с 1080p это с запасом (замер: 4 логических ядра
# ноутбука — 70 к/с выхода 1080×1920).
ENC_THREADS = max(2, min(8, CPU_BUDGET - 2))


# --------------------------------------------------------------------- движок
def _load():
    """Один раз на воркер. Холодный старт = загрузка модулей, не скачивание."""
    if _PIPES["t2i"] is not None:
        return _PIPES["t2i"], _PIPES["i2i"]

    from diffusers import (LCMScheduler, StableDiffusionImg2ImgPipeline,
                           StableDiffusionPipeline)

    # ⚠️⚠️ `variant` обязан совпадать с тем, ЧТО ЗАПЕЧЕНО в образ. Мы кладём
    # fp16-файлы (`*.fp16.safetensors`, вдвое легче), а загрузчик без этого
    # параметра ищет обычные имена, не находит, откатывается к `.bin`, не
    # находит и падает: «Error no file named diffusion_pytorch_model.bin found
    # in directory /models/sd15/unet». Локально дефект НЕ воспроизводится —
    # там кэш HF с обычными файлами, — и виден только живым прогоном образа
    # (замер 04.09: 19.6 минуты холодного старта до этой ошибки).
    # Меняешь `variant` — меняй allow_patterns в download_models.py, это пара.
    t2i = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, variant=MODEL_VARIANT,
        safety_checker=None, requires_safety_checker=False)
    t2i.scheduler = LCMScheduler.from_config(t2i.scheduler.config)
    # ⚠️ Порядок: сплавление LoRA на CPU в fp16 идёт МИНУТАМИ при нулевой
    # загрузке GPU и выглядит как зависание. На GPU — доли секунды.
    t2i.to("cuda")
    t2i.load_lora_weights(LORA_DIR)
    t2i.fuse_lora()
    t2i.unload_lora_weights()      # веса вплавлены, копия адаптера только ест VRAM
    t2i.set_progress_bar_config(disable=True)

    i2i = StableDiffusionImg2ImgPipeline(**t2i.components,
                                         requires_safety_checker=False)
    i2i.set_progress_bar_config(disable=True)

    _PIPES["t2i"], _PIPES["i2i"] = t2i, i2i
    return t2i, i2i


def _steps_for(strength: float) -> int:
    """Держим ЭФФЕКТИВНЫЕ шаги: img2img запускает только долю strength от них.
    Формула живёт в motion.steps_for — кривые движения считают шаги той же."""
    return motion.steps_for(strength, EFF_STEPS)


def _cover(img, w, h):
    """Кроп «cover» под кадр: масштаб по большей стороне, срез по центру.
    Растянуть нельзя — обложка узнаваема только с сохранённой пропорцией."""
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    nw, nh = max(w, int(round(sw * scale))), max(h, int(round(sh * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def _warp(img, zoom, angle, tx, ty):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT_101)


def _mid_frame(prev, cur, zoom, angle, tx, ty):
    """Кадр ровно посередине шага петли — для `fps_out` = 2 × fps.

    Мы ЗНАЕМ, какой аффинкой кадр cur получен из prev (её же строил `_warp`),
    поэтому середина — это prev, сдвинутый на «половину» шага (A, где A∘A = M),
    и cur, отведённый назад на ту же половину, смешанные пополам. Геометрия
    обеих половин совпадает, и смешение — это смена текстуры img2img, а не
    двойной контур.

    ⚠️⚠️ Почему не фильтр ffmpeg (замер 24.10 на настоящих кадрах петли, README):
      · minterpolate mci — ~8.8 к/с входа на ОДНОМ ядре ноутбука i9 (фильтр
        однопоточный), то есть на ядре сервера он медленнее генерации (~6 к/с)
        и тормозил бы весь прогон; к тому же теряет кадры (189 из 192);
      · framerate (смешение) — двойной контур на каждом втором кадре;
      · оба фильтра на тёмной склейке (неон → неон) НЕ видят смену сцены и
        рисуют кашу из двух сцен на месте жёсткой склейки.
    Здесь: 1 мс на кадр, точность выше mci (PSNR +0.2…0.35 дБ, когда кадр
    петли выброшен и восстановлен), а склейки знаем из плана, а не угадываем.
    INTER_LINEAR, а не LANCZOS: середина — смесь двух кадров, резкость ей не
    прибавляется, а мерка с LANCZOS дала даже чуть хуже при 7× цене.
    """
    h, w = prev.shape[:2]
    a = np.array(motion.half_affine(w, h, zoom, angle, tx, ty), dtype=np.float64)
    fwd = cv2.warpAffine(prev, a, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    bwd = cv2.warpAffine(cur, a, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT_101)
    return cv2.addWeighted(fwd, 0.5, bwd, 0.5, 0)


def _lab_stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return [(lab[:, :, c].mean(), lab[:, :, c].std()) for c in range(3)]


def _color_match(img, ref):
    """Выравнивание к первому кадру ШОТА — без него палитра уходит в сепию."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = np.empty_like(lab)
    for c in range(3):
        s_mean, s_std = lab[:, :, c].mean(), lab[:, :, c].std() + 1e-6
        t_mean, t_std = ref[c]
        out[:, :, c] = (lab[:, :, c] - s_mean) * (t_std / s_std) + t_mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


class _Encoder:
    """Потоковый кодировщик: кадр уходит в ffmpeg СРАЗУ, в памяти не копится.

    ⚠️⚠️ Копить кадры и кодировать в конце — это то, что работает на тесте и
    умирает на цели. Замер: 180 кадров 512×512 это 141 МБ, а клип 3 мин при
    12 fps — 2160 кадров, то есть **1.7 ГБ живых байт**, 5 мин — 2.8 ГБ. Здесь
    расход постоянный, и кодирование идёт ПАРАЛЛЕЛЬНО генерации (ffmpeg успевает
    за диффузией с огромным запасом: 0.6 с на кадр против единиц миллисекунд).

    Пометка ИИ живёт ЗДЕСЬ — см. раздел про ст. 50(2) в README. Поля те же, что
    пишет public/js/ai-mark.js: ffmpeg маппит `comment` в `©cmt` внутри
    moov>udta>meta>ilst, то есть в тот же атом.
    """

    def __init__(self, w, h, fps, mark, out=None, threads=None):
        """`out` = (W, H) — во что масштабировать готовый кадр, или None.
        `threads` — потолок потоков x264 (None — как раньше, решает ffmpeg).

        ⚠️ Масштабирование делает ffmpeg, а не диффузия: рисовать 720×1280 SD 1.5
        не умеет (дублирует объекты), а площадки требуют именно такого размера —
        Spotify Canvas для 9:16 документирует 720×1280 и 1080×1920. Апскейл lanczos на выходе
        стоит доли процента времени прогона и не трогает петлю.
        ⚠️ Пропорция сохраняется точно: `out` берётся из config/ai-video.php,
        где стороны подобраны так, что 432×768 → 720×1280 это ровно ×1.666.
        """
        self.path = tempfile.mktemp(suffix=".mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "pipe:"]
        if out and (int(out[0]), int(out[1])) != (int(w), int(h)):
            cmd += ["-vf", f"scale={int(out[0])}:{int(out[1])}:flags=lanczos"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
                "-crf", "18", "-preset", "veryfast"]
        if out and int(out[0]) * int(out[1]) >= BIG_OUT_AREA:
            # ⚠️⚠️ Потолок битрейта только для выхода от 1080², и вот почему
            # (замер 24.10 на настоящих кадрах петли, README): 1080×1920 при
            # crf 18 — ~10 Мбит/с, 5 минут ≈ 377 МБ, а приём файла у приложения
            # — 480 МБ (CLIP_UPLOAD_MAX_MB; ⚠️ до 29.10 там стояли 200, и этот
            # комментарий утверждал 512 — README «Доставка файла»). Плотная текстура (ролик «масло» в 1.8 раза тяжелее медианы)
            # или шум пробивали бы этот потолок ПОСЛЕ 13 минут GPU. capped CRF:
            # обычный клип потолка почти не касается (10.02 → 9.88 Мбит/с), худший
            # прижат к ~10.3 Мбит/с → 5 минут ≤ ~390 МБ. 720×1280 и меньше —
            # побайтно как раньше: там 5 минут ≈ 190 МБ.
            cmd += ["-maxrate", BIG_OUT_MAXRATE, "-bufsize", BIG_OUT_BUFSIZE]
        if threads:
            # Только у новых выходов (2·fps, от 1080²): число потоков x264 меняет
            # байты потока, а прежние выходы обязаны остаться побайтно прежними.
            cmd += ["-threads", str(int(threads))]
        cmd += ["-movflags", "+faststart"]
        if mark:
            cmd += ["-metadata", "comment=AI-generated video (Timbrica clip generator)",
                    "-metadata", "synopsis=digitalsourcetype=trainedAlgorithmicMedia",
                    "-metadata", "encoder_tool=Timbrica AI tools"]
        cmd += [self.path]
        # ⚠️ Кодировщик уступает ядро петле: GPU простаивает, пока Python не
        # подал следующий кадр, а ffmpeg догонит в паузах диффузии. `nice`
        # префиксом, а не preexec_fn: тот небезопасен при живых потоках, а
        # у runpod-обвязки они есть всегда. На байты потока не влияет.
        if os.name == "posix" and shutil.which("nice"):
            cmd = ["nice", "-n", "10"] + cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        # ⚠️⚠️ Кадры в трубу пишет ОТДЕЛЬНЫЙ поток через ограниченную очередь.
        # Труба — 64 КБ, и при прямой записи петля стояла каждый раз, когда x264
        # набирал кадры пачкой (lookahead, потоки кадров): живой A/B 24.10 на
        # 4090 — 24 к/с 1080p удлиняли кадр петли на 7–37 %. Очередь держит
        # QUEUE_FRAMES кадров кадра генерации (~1 МБ каждый при 432×768) —
        # память ограничена, а рывки кодировщика больше не стопорят GPU.
        self.t_blocked = 0.0
        self._err = None
        self._q = queue.Queue(maxsize=QUEUE_FRAMES)
        self._pump = threading.Thread(target=self._drain_to_pipe, daemon=True)
        self._pump.start()

    def _drain_to_pipe(self):
        try:
            while True:
                f = self._q.get()
                if f is None:
                    return
                self.proc.stdin.write(f.tobytes())
        except Exception as exc:                 # ffmpeg умер — труба закрыта
            self._err = exc
            while self._q.get() is not None:     # не держим петлю на полной очереди
                pass

    def write(self, frame):
        if self._err is not None:
            raise RuntimeError(f"encode_pipe: {type(self._err).__name__}")
        t = time.time()
        self._q.put(frame)          # кадр больше не меняется: копия не нужна
        self.t_blocked += time.time() - t

    def finish(self):
        self._q.put(None)
        self._pump.join()
        if self._err is not None:
            # Причина — в stderr ffmpeg, а не в BrokenPipe: её и показываем.
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass
            rc = self.proc.wait(timeout=60)
            raise RuntimeError(f"encode_failed_{rc}: {self.proc.stderr.read()[-300:]!r}")
        self.proc.stdin.close()
        rc = self.proc.wait(timeout=300)
        if rc != 0:
            raise RuntimeError(f"encode_failed_{rc}: {self.proc.stderr.read()[-300:]!r}")
        with open(self.path, "rb") as f:
            data = f.read()
        if len(data) < 10_000 or data[4:8] != b"ftyp":
            raise RuntimeError("encode_bad_output")
        return data

    def close(self):
        """Уборка на любом пути выхода — ни ffmpeg, ни поток-насос не должны
        пережить хендлер: воркер живёт десятки заданий, и утёкший поток на
        каждом упавшем — это утечка, которую видно только через сутки."""
        try:
            if self.proc.poll() is None:
                self.proc.kill()        # насос, пишущий в трубу, получит ошибку
                self.proc.wait(timeout=10)
        except Exception:
            pass
        try:
            self._q.put(None, timeout=5)    # после finish() лишний None безвреден
        except Exception:
            pass
        self._pump.join(timeout=5)
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass


def _schedule(total, fps, shot_len, plan, base_prompt):
    """Кадр → (начало ли шота, промпт этого шота).

    Без плана шоты идут равномерно по `shot_len` и все с одним промптом. С планом
    длину и текст каждого шота задаёт приложение — это и есть «расписание
    промптов по таймлайну», которым Neural Frames рулит сценами.

    ⚠️ План обрезается по длине клипа, а не наоборот: оплачена длина, и лишний
    шот в плане не имеет права её продлить. Если план короче клипа, последний
    шот доигрывает остаток — это лучше, чем оборвать выданный результат.
    """
    prompts = [base_prompt] * total
    starts = set()

    if not plan:
        for i in range(0, total, shot_len):
            starts.add(i)
        return starts, prompts

    at = 0
    for start, length, shot in _plan_walk(total, fps, shot_len, plan):
        text = (shot.get("prompt") or base_prompt).strip() or base_prompt
        starts.add(start)
        for i in range(start, min(start + length, total)):
            prompts[i] = text
        at = start + length

    if at < total:                      # план короче клипа — добиваем последним
        for i in range(at, total):
            prompts[i] = prompts[at - 1] if at > 0 else base_prompt

    starts.add(0)
    return starts, prompts


def _plan_walk(total, fps, shot_len, plan):
    """(начало, длина, шот) каждого шота плана, что помещается в клип, — ОДИН
    обход на расписание (_schedule) и на сиды сцен (_shot_seeds): разойдись они
    хоть на кадр — раскадровка показала бы не те кадры, с которых начнётся клип."""
    at = 0
    for shot in plan:
        if at >= total:
            break
        try:
            secs = float(shot.get("seconds") or 0)
        except (TypeError, ValueError):
            secs = 0.0
        length = max(int(round(secs * fps)), 2) if secs > 0 else shot_len
        yield at, length, shot
        at += length


def _shot_seeds(total, fps, shot_len, plan):
    """Кадр начала шота -> свой сид шота из плана (раскадровка 29.10.2026).

    Человек перерисовал кадр одной сцены в раскадровке — у неё свой сид, и клип
    обязан начать эту сцену с того же кадра. Шот без своего сида — прежний
    `seed + номер * 7919`, побайтно как раньше."""
    out = {}
    for start, _length, shot in _plan_walk(total, fps, shot_len, plan or []):
        s = shot.get("seed")
        if isinstance(s, int) and not isinstance(s, bool) and 0 <= s <= 0x7FFFFFFF:
            out[start] = s
    return out


def _thumb_b64(img, long_side=384, quality=82):
    """Кадр раскадровки -> JPEG base64. Уменьшенный: 43 сцены в одном ответе
    RunPod (~0.5 МБ), а не 43 полноразмерных кадра."""
    w, h = img.size
    k = min(1.0, long_side / max(w, h))
    if k < 1.0:
        img = img.resize((max(16, round(w * k)), max(16, round(h * k))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _shot_first_frame(t2i, i2i, text, init_img, init_strength, width, height, g):
    """Первый кадр сцены — ОДИН код на клип и на раскадровку."""
    if init_img is not None:
        # Сцена стартует с обложки, перерисованной под её промпт. Шаги —
        # эффективные (см. _steps_for): при силе 0.58 и четырёх номинальных
        # шагах модель получила бы два.
        return i2i(prompt=text, negative_prompt=NEG, image=init_img, strength=init_strength,
                   num_inference_steps=_steps_for(init_strength),
                   guidance_scale=1.5, generator=g).images[0]
    return t2i(prompt=text, negative_prompt=NEG, num_inference_steps=4,
               guidance_scale=1.5, width=width, height=height, generator=g).images[0]


def _gpu_name():
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — метрика не роняет прогон
        return None


def _render_range(shot_starts, total, render_shot):
    """Какие кадры рисовать: [0, total) — весь клип, или кадры сцены
    `render_shot` (с 1) — от её начала до начала следующей. (None, None) —
    такой сцены в расписании нет."""
    if not render_shot:
        return 0, total
    starts = sorted(shot_starts)
    if not (1 <= render_shot <= len(starts)):
        return None, None
    lo = starts[render_shot - 1]
    hi = starts[render_shot] if render_shot < len(starts) else total
    return lo, hi


def _deliver(mp4, upload, post=None, sleep=None, clock=None):
    """Отдать готовый файл приложению; вернуть расписку.

    ⚠️⚠️ ПОЧЕМУ НЕ ИНЛАЙН: замер 03.09 — 30 секунд видео это 7.9 МБ, значит
    3 минуты ≈ 47 МБ, 5 минут ≈ 78 МБ, плюс треть на base64. Это выше потолка
    результата RunPod, то есть длинный клип инлайном просто не доедет.
    Короткий (превью) по-прежнему едет инлайном — ему дверь не нужна.

    ⚠️⚠️ ПОЧЕМУ ПОРЦИЯМИ И С ПОВТОРОМ (29.10.2026). До этого файл уходил одним
    POST и один обрыв выбрасывал всю отрисовку. 28.10 18:20–19:25 сеть до
    Бегета зависала (nginx отвечал 408 после 60 с тишины каждой пятидесятой
    порции чужих загрузок), и три клипа 1080p по 226 с (~220 МБ) упали
    `upload_failed: ConnectionError` — каждый после десяти минут GPU. Теперь,
    если приложение назвало `chunk_bytes`, файл идёт порциями протоколом
    смещений (ClipUploadController::storeSlice): ответ сервера `received` —
    единственная правда о том, с какого байта продолжать, а отказ стоит одной
    порции. Сдаёмся, только когда продвижения нет `stall_s` секунд или вся
    загрузка идёт дольше `total_s` — у задания RunPod свой потолок (30 мин),
    и пятиминутный 1080p сам занимает его большую часть.

    Расписка несёт размер и sha256, и приложение сверяет её с тем, что реально
    легло на диск: «доехало битым» обязано быть отличимо от «доехало».
    """
    # Импорты здесь, а не только наверху: upload_selftest.py берёт эти функции
    # из исходника без torch и без модульных имён.
    import hashlib
    import os
    import time

    url = (upload.get("url") or "").strip()
    if not url:
        return {"error": "upload_no_url"}
    if post is None:
        import requests
        post = requests.post

    digest = hashlib.sha256(mp4).hexdigest()
    headers = {
        "X-Clip-Gen": str(upload.get("gen_id") or ""),
        "X-Clip-Expires": str(upload.get("expires") or ""),
        "X-Clip-Sig": str(upload.get("sig") or ""),
    }
    way = dict(post=post, sleep=sleep or time.sleep, clock=clock or time.monotonic,
               stall_s=float(os.environ.get("UPLOAD_STALL_S", 360)),
               total_s=float(os.environ.get("UPLOAD_TOTAL_S", 900)))
    chunk = int(upload.get("chunk_bytes") or 0)
    if chunk > 0:
        return _deliver_slices(url, headers, mp4, digest, chunk, **way)
    return _deliver_whole(url, headers, mp4, digest, **way)


def _upload_retryable(status):
    """Отказ, который стоит повторить: обрыв (0), 408 тишины, перегрузка, 5xx.
    403 подписи, 404/409 улаженной генерации, 413 размера, 422 не того файла —
    повтор их не изменит."""
    return status == 0 or status in (408, 425, 429, 500, 502, 503, 504)


def _upload_pause(attempt):
    """Пауза перед повтором: быстро на одиночной заминке, не чаще раза в 30 с
    в долгом провале сети."""
    return (2, 5, 10, 20)[attempt] if attempt < 4 else 30


def _upload_send(post, url, headers, timeout, **kw):
    """Один запрос: (код, json, текст ошибки для лога). Обрыв — код 0."""
    try:
        resp = post(url, headers=headers, timeout=timeout, **kw)
    except Exception as exc:
        return 0, {}, f"upload_failed: {type(exc).__name__}"[:200]
    try:
        body = resp.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    return resp.status_code, body, f"upload_http_{resp.status_code}: {(resp.text or '')[:150]}"


def _deliver_slices(url, headers, mp4, digest, chunk, post, sleep, clock, stall_s, total_s):
    size = len(mp4)
    offset = 0
    started = progressed = clock()
    attempt = retries = 0
    while True:
        status, body, err = _upload_send(
            post, url, headers, (15, 120),
            data={"offset": str(offset), "size": str(size), "sha256": digest},
            files={"chunk": ("clip.part", mp4[offset:offset + chunk], "application/octet-stream")},
        )
        if status == 200 and body.get("done"):
            return {"stored": True, "bytes": size, "sha256": digest, "upload_retries": retries}
        # Сервер называет, сколько у него есть: и после принятой порции (200),
        # и после порции не с того места (409 gap) — продолжать с этого байта.
        if (status == 200 or (status == 409 and body.get("error") == "gap")) and "received" in body:
            received = int(body["received"])
            moved = received > offset
            offset = max(0, min(received, size))
            if moved:
                progressed = clock()
                attempt = 0
                continue
        elif not _upload_retryable(status):
            return {"error": err}
        now = clock()
        if now - progressed >= stall_s or now - started >= total_s:
            return {"error": err}
        sleep(_upload_pause(attempt))
        attempt += 1
        retries += 1


def _deliver_whole(url, headers, mp4, digest, post, sleep, clock, stall_s, total_s):
    """Приложение до 29.10 порций не знает — файл целиком, с тем же повтором."""
    started = clock()
    attempt = 0
    while True:
        status, body, err = _upload_send(
            post, url, headers, (15, 300),
            files={"file": ("clip.mp4", mp4, "video/mp4")},
        )
        if status == 200:
            return {"stored": True, "bytes": len(mp4), "sha256": digest, "upload_retries": attempt}
        # Текст ответа полезен в логе прогона: 403 подписи и 413 размера —
        # разные аварии, и различать их надо на этой стороне тоже.
        if not _upload_retryable(status) or attempt >= 2 or clock() - started >= total_s:
            return {"error": err}
        sleep(_upload_pause(attempt))
        attempt += 1


# -------------------------------------------------------------------- handler
def handler(event):
    inp = event.get("input") or {}

    prompt = (inp.get("prompt") or "").strip()
    if not prompt or len(prompt) > 2000:
        return {"error": "bad_prompt"}

    style_key = inp.get("style") or "flow"
    if style_key not in STYLES:
        return {"error": "bad_style"}
    mo = dict(STYLES[style_key])

    try:
        seconds = float(inp.get("seconds", 8))
        fps = int(inp.get("fps", 12))
        shot_seconds = float(inp.get("shot_seconds", 8))
        # ⚠️ `size` (квадрат) остаётся ради совместимости: провайдер шлёт
        # width/height, но старый вызов не имеет права сломаться.
        size = int(inp.get("size", 512))
        width = int(inp.get("width", size))
        height = int(inp.get("height", size))
        out_size = inp.get("out")
        out_size = ((int(out_size[0]), int(out_size[1]))
                    if isinstance(out_size, (list, tuple)) and len(out_size) == 2 else None)
    except (TypeError, ValueError):
        return {"error": "bad_params"}

    if not (1 <= seconds <= MAX_SECONDS):
        return {"error": "bad_seconds"}
    if not (8 <= fps <= 15) or not (4 <= shot_seconds <= 15):
        return {"error": "bad_params"}
    # ⚠️⚠️ Белый список, а не диапазон. Стороны обязаны быть кратны 16 (иначе
    # VAE роняет размерность) и не уходить от 512 дальше одной ступени: на
    # 576×1024 SD 1.5 начинает дублировать объекты — «две головы» вместо кадра.
    # Здесь стоят ровно те пары, что объявлены в config/ai-video.php → aspects;
    # расхождение ловит ClipGeneratorLaneTest.
    if (width, height) not in ALLOWED_SIZES:
        return {"error": "bad_params"}
    if out_size is not None and not (64 <= out_size[0] <= 2048 and 64 <= out_size[1] <= 2048):
        return {"error": "bad_params"}

    # Стартовая картинка (обложка альбома, фото). Каждая сцена начинается с НЕЁ,
    # перерисованной под промпт и вид, — а не с кадра, нарисованного по тексту.
    # Для музыканта это главный сценарий: у каждого релиза есть обложка.
    # ⚠️ Кроп под кадр здесь, а не в приложении: приложение уменьшает до ~768 px
    # ради веса запроса, а точную форму знает только эта пара width×height.
    init_img = None
    b64 = inp.get("init_image_b64")
    if b64:
        try:
            raw = base64.b64decode(b64, validate=True)
            if len(raw) > 6 * 1024 * 1024:
                return {"error": "bad_image"}
            init_img = _cover(Image.open(io.BytesIO(raw)).convert("RGB"), width, height)
        except Exception:
            return {"error": "bad_image"}
    try:
        init_strength = float(inp.get("init_strength", 0.58))
    except (TypeError, ValueError):
        return {"error": "bad_params"}
    # Ниже 0.35 картинка почти не меняется (нет движения между сценами), выше
    # 0.8 обложка узнаётся только по цвету — то есть теряется смысл входа.
    init_strength = max(0.35, min(0.80, init_strength))

    if inp.get("strength") is not None:
        try:
            s = float(inp["strength"])
        except (TypeError, ValueError):
            return {"error": "bad_params"}
        if not (0.30 <= s <= 0.60):
            return {"error": "bad_params"}
        mo["strength"] = s

    mark = bool(inp.get("mark", True))
    seed = inp.get("seed")
    seed = int(seed) if seed is not None else int(time.time()) & 0x7FFFFFFF

    total = max(int(round(seconds * fps)), 2)
    shot_len = max(int(round(shot_seconds * fps)), 2)
    steps = _steps_for(mo["strength"])

    # Частота ВЫДАННОГО файла: fps (как было) или 2 × fps — середины шагов
    # (_mid_frame). Генерация при этом та же: n_frames и цена прогона не
    # меняются, меняется только плавность. Другие значения — не «округлим»,
    # а отказ: молча выдать 12 к/с тому, кто просил 24, хуже ошибки.
    try:
        fps_out = int(inp.get("fps_out") or fps)
    except (TypeError, ValueError):
        return {"error": "bad_params"}
    if fps_out not in (fps, 2 * fps):
        return {"error": "bad_params"}
    double = fps_out == 2 * fps

    # Пульс под музыку. Без `react` поля energy/hits не читаются вовсе:
    # старый вызов не может сломаться о поле, которого он не слал.
    try:
        react = motion.parse_react(inp.get("react"))
        energy = motion.parse_energy(inp.get("energy"), total) if react > 0 else None
        hits = motion.parse_hits(inp.get("hits"), total) if react > 0 else []
    except ValueError:
        return {"error": "bad_params"}

    # Пересъёмка ОДНОЙ сцены (26.10.2026): приложение шлёт задание клипа
    # целиком и номер сцены (с 1), которую нарисовать. Расписание и кривая
    # пульса строятся по ВСЕМУ клипу — сцена выходит ровно той длины и с тем
    # же движением, что в клипе, и приложение вклеивает её на место
    # (App\Services\AiVideo\ClipSplice). shot_seed — свой сид этой сцены:
    # иначе пересъёмка повторила бы её кадр в кадр.
    try:
        render_shot = int(inp.get("render_shot") or 0)
        shot_seed = inp.get("shot_seed")
        shot_seed = int(shot_seed) if shot_seed is not None else None
    except (TypeError, ValueError):
        return {"error": "bad_params"}

    plan = inp.get("shots")
    if plan is not None and not isinstance(plan, list):
        return {"error": "bad_shots"}
    if isinstance(plan, list) and len(plan) > 200:
        return {"error": "too_many_shots"}
    shot_starts, frame_prompts = _schedule(
        total, fps, shot_len,
        [s for s in (plan or []) if isinstance(s, dict)],
        prompt,
    )

    plan_seeds = _shot_seeds(total, fps, shot_len, [s for s in (plan or []) if isinstance(s, dict)])

    # Раскадровка (29.10.2026): только ПЕРВЫЙ кадр каждой сцены — тот же, с
    # которого начнётся клип с этим сидом (тот же промпт, сид, размер, обложка),
    # без петли warp и без файла. `only` — номера сцен (с 1), если перерисовать
    # надо не все. Человек видит, что получит, ДО оплаты клипа.
    if inp.get("storyboard"):
        only = inp.get("only")
        if only is not None:
            if not isinstance(only, list) or len(only) > 200 or not all(
                    isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in only):
                return {"error": "bad_params"}
            only = set(only)
        t2i, i2i = _load()
        t0 = time.time()
        stills = []
        try:
            for j, start in enumerate(sorted(shot_starts), start=1):
                if only is not None and j not in only:
                    continue
                s_j = plan_seeds.get(start, seed + j * 7919)
                g = torch.Generator("cuda").manual_seed(s_j)
                img = _shot_first_frame(t2i, i2i, frame_prompts[start], init_img, init_strength,
                                        width, height, g)
                stills.append({"shot": j, "frame": start, "seed": s_j, "jpg": _thumb_b64(img)})
        except torch.cuda.OutOfMemoryError:
            return {"error": "out_of_memory"}
        except Exception as exc:                  # приложение вернёт токены целиком
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        return {"storyboard": stills, "shots": len(shot_starts), "seed": seed,
                "width": width, "height": height, "gpu": _gpu_name(),
                "timings": {"first_frames": round(time.time() - t0, 2)}}

    lo, hi = _render_range(shot_starts, total, render_shot)
    if lo is None:
        return {"error": "bad_render_shot"}

    # Параметры перехода (i-1 → i) по кадрам. react = 0 — ровно константы
    # стиля (те же объекты float), то есть побайтно прежний ролик.
    curves = motion.motion_curves(total, fps, mo, react, energy, hits,
                                  shot_starts, EFF_STEPS)

    t2i, i2i = _load()
    t_first = t_loop = t_interp = 0.0
    ref = None
    per_frame = []
    shots = 0
    frames_out = 0

    # Новые выходы (2·fps или от 1080²) — с потолком потоков x264; прежние —
    # прежней командой, побайтно.
    big = out_size is not None and out_size[0] * out_size[1] >= BIG_OUT_AREA
    enc_threads = ENC_THREADS if (double or big) else None

    # Номер шота считается от начала клипа: у пересъёмки сцена k — k-я, как в
    # клипе (без shot_seed сид тот же, что был у неё в клипе).
    if render_shot:
        shots = render_shot - 1

    t0 = time.time()
    enc = _Encoder(width, height, fps_out, mark, out_size, threads=enc_threads)
    try:
        for i in range(lo, hi):
            # Начало шота: композиция ставится ЗАНОВО. Без этого зум за минуты
            # уводит камеру внутрь детали — см. README, пункт 1.
            if i in shot_starts:
                # У пересъёмки повтор перед её первым кадром — кадр ЧУЖОЙ
                # (предыдущей сцены): он остаётся в клипе, не в сцене.
                if double and i > lo:
                    # Середина на склейке — ПОВТОР последнего кадра шота, а не
                    # смесь двух сцен: склейка обязана остаться жёсткой и
                    # стоять на том же моменте, что при 12 к/с.
                    enc.write(cur)
                    frames_out += 1
                shots += 1
                # Сид сцены: пересъёмка — свой; сцена, перерисованная в
                # раскадровке, — свой из плана; иначе прежний seed + номер.
                g = torch.Generator("cuda").manual_seed(
                    shot_seed if (render_shot and shot_seed is not None)
                    else plan_seeds.get(i, seed + shots * 7919))
                ts = time.time()
                img = _shot_first_frame(t2i, i2i, frame_prompts[i], init_img, init_strength,
                                        width, height, g)
                t_first += time.time() - ts
                cur = np.array(img)
                ref = _lab_stats(cur)          # цветовой якорь СВОЙ у каждого шота
                enc.write(cur)
                frames_out += 1
                continue

            ts = time.time()
            prev = cur
            zoom, angle = curves["zoom"][i], curves["angle"][i]
            tx, ty = curves["tx"][i], curves["ty"][i]
            moved = _warp(prev, zoom, angle, tx, ty)
            out = i2i(prompt=frame_prompts[i], negative_prompt=NEG,
                      image=Image.fromarray(moved), strength=curves["strength"][i],
                      num_inference_steps=curves["steps"][i], guidance_scale=1.5,
                      generator=g).images[0]
            cur = _color_match(np.array(out), ref)
            if double:
                tm = time.time()
                enc.write(_mid_frame(prev, cur, zoom, angle, tx, ty))
                frames_out += 1
                t_interp += time.time() - tm
            enc.write(cur)
            frames_out += 1
            dt = time.time() - ts
            t_loop += dt
            per_frame.append(dt)

        if double:
            # Последний кадр держится на свою вторую половину шага: 2·N кадров
            # при 2·fps — ровно та же длительность, что N при fps. У пересъёмки
            # это тот же кадр, что в клипе стоит перед следующей сценой (или в
            # конце): её выход — ровно кадры [2·lo, 2·hi) клипа.
            enc.write(cur)
            frames_out += 1

        ts = time.time()
        mp4 = enc.finish()
        t_encode = time.time() - ts
    except torch.cuda.OutOfMemoryError:
        return {"error": "out_of_memory"}
    except Exception as exc:                      # приложение вернёт токены целиком
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        enc.close()
        gc.collect()
        torch.cuda.empty_cache()

    gen = time.time() - t0
    metrics = {
        # Размер ВЫДАННОГО файла, а не рисованного кадра: при апскейле это
        # разные числа, и приложение подписывает результат по этому полю.
        "width": (out_size[0] if out_size else width), "height": (out_size[1] if out_size else height),
        "render_width": width, "render_height": height, "fps": fps,
        # Сид уезжает наружу ради «того же образа»: без него повторный прогон
        # рисует другой мир, и клип нельзя доработать, только переиграть.
        "seed": seed,
        # n_frames — СГЕНЕРИРОВАННЫЕ кадры (на них считается цена и время);
        # в файле их frames_out: при fps_out = 2·fps вдвое больше.
        "n_frames": hi - lo, "seconds": round((hi - lo) / fps, 2), "shots": shots,
        "gen_seconds": round(gen, 1),
        "sec_per_frame": round(float(np.median(per_frame)), 3) if per_frame else None,
        "steps_per_frame": steps,
        # Эхо принятого — им приложение отличает новый образ от старого:
        # образ до 24.10 этих полей не вернёт, и «просили 24, а пришло 12»
        # видно по ответу, а не по жалобе.
        "fps_out": fps_out, "frames_out": frames_out,
        "react": react, "hits_used": len(hits),
        # Эхо пересъёмки: приложение сверяет его с заказанной сценой — образ
        # без render_shot молча нарисовал бы весь клип по цене одной сцены.
        "render_shot": render_shot, "frames": [lo, hi],
        # write — сколько петля простояла на полной очереди к кодировщику:
        # ненулевое значение = кодировщик не успевает и тормозит GPU.
        "timings": {"first_frames": round(t_first, 1), "loop": round(t_loop, 1),
                    "encode": round(t_encode, 1), "interp": round(t_interp, 1),
                    "write": round(enc.t_blocked, 1)},
        "cpu": dict(CPU_INFO, enc_threads=enc_threads),
        # Какая карта рисовала. 24–25.10 клипы шли 0.27–0.29 с/кадр, с 26.10 —
        # 0.21 при том же коде; без имени карты это нельзя ни доказать, ни
        # учесть в цене (пулы ADA_24 и AMPERE_24 разнятся вдвое по цене часа).
        "gpu": _gpu_name(),
    }

    upload = inp.get("upload")
    if isinstance(upload, dict) and upload.get("url"):
        # Длинный клип: файл едет приложению, инлайном — только расписка.
        receipt = _deliver(mp4, upload)
        if "error" in receipt:
            return receipt          # приложение вернёт токены целиком
        return {**receipt, **metrics}

    # Короткий прогон (превью шота) — инлайном, дверь ему не нужна.
    # ⚠️ Потолок сознательный: у RunPod есть предел результата, и молча упереться
    # в него хуже, чем назвать причину.
    if len(mp4) > INLINE_MAX_BYTES:
        return {"error": "result_too_large_for_inline"}
    return {"mp4_b64": base64.b64encode(mp4).decode(), **metrics}


runpod.serverless.start({"handler": handler})
