"""RunPod serverless handler — видеоклип по промпту через warp-петлю SD 1.5 + LCM.

Контракт и обоснования — README.md рядом. Три вещи, которые здесь выглядят как
мелочи, а на самом деле держат результат (все три из замера 2026-09-03):
  · склейки каждые shot_seconds — иначе композиция уползает и клип невозможен;
  · коррекция цвета к первому кадру ШОТА — иначе палитра уходит в сепию;
  · эффективные шаги = steps × strength — иначе картинка постеризуется.
"""
import base64
import gc
import hashlib
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
import runpod
import torch
from PIL import Image

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

# Потолок инлайн-выдачи. Ниже настоящего предела RunPod с запасом: base64
# раздувает байты на треть, и упереться в предел ПОСЛЕ прогона — значит выбросить
# уже потраченные GPU-минуты.
INLINE_MAX_BYTES = int(os.environ.get("INLINE_MAX_BYTES", 6 * 1024 * 1024))
EFF_STEPS = float(os.environ.get("EFF_STEPS", 3.5))

_PIPES = {"t2i": None, "i2i": None}


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
    """Держим ЭФФЕКТИВНЫЕ шаги: img2img запускает только долю strength от них."""
    return max(int(round(EFF_STEPS / max(strength, 0.05))), 4)


def _warp(img, zoom, angle, tx, ty):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT_101)


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

    def __init__(self, w, h, fps, mark):
        self.path = tempfile.mktemp(suffix=".mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "pipe:",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
               "-crf", "18", "-preset", "veryfast", "-movflags", "+faststart"]
        if mark:
            cmd += ["-metadata", "comment=AI-generated video (Timbrica clip generator)",
                    "-metadata", "synopsis=digitalsourcetype=trainedAlgorithmicMedia",
                    "-metadata", "encoder_tool=Timbrica AI tools"]
        cmd += [self.path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def finish(self):
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
        """Уборка на любом пути выхода — ffmpeg не должен пережить хендлер."""
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.close()
                except (OSError, ValueError):
                    pass
                self.proc.kill()
                self.proc.wait(timeout=10)
        except Exception:
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
    for shot in plan:
        if at >= total:
            break
        text = (shot.get("prompt") or base_prompt).strip() or base_prompt
        try:
            secs = float(shot.get("seconds") or 0)
        except (TypeError, ValueError):
            secs = 0.0
        length = max(int(round(secs * fps)), 2) if secs > 0 else shot_len
        starts.add(at)
        for i in range(at, min(at + length, total)):
            prompts[i] = text
        at += length

    if at < total:                      # план короче клипа — добиваем последним
        for i in range(at, total):
            prompts[i] = prompts[at - 1] if at > 0 else base_prompt

    starts.add(0)
    return starts, prompts


def _deliver(mp4, upload):
    """Отдать готовый файл приложению; вернуть расписку.

    ⚠️⚠️ ПОЧЕМУ НЕ ИНЛАЙН: замер 03.09 — 30 секунд видео это 7.9 МБ, значит
    3 минуты ≈ 47 МБ, 5 минут ≈ 78 МБ, плюс треть на base64. Это выше потолка
    результата RunPod, то есть длинный клип инлайном просто не доедет.
    Короткий (превью) по-прежнему едет инлайном — ему дверь не нужна.

    Расписка несёт размер и sha256, и приложение сверяет её с тем, что реально
    легло на диск: «доехало битым» обязано быть отличимо от «доехало».
    """
    import requests

    url = (upload.get("url") or "").strip()
    if not url:
        return {"error": "upload_no_url"}

    digest = hashlib.sha256(mp4).hexdigest()
    headers = {
        "X-Clip-Gen": str(upload.get("gen_id") or ""),
        "X-Clip-Expires": str(upload.get("expires") or ""),
        "X-Clip-Sig": str(upload.get("sig") or ""),
    }
    try:
        resp = requests.post(
            url, headers=headers,
            files={"file": ("clip.mp4", mp4, "video/mp4")},
            timeout=(15, 300),
        )
    except Exception as exc:
        return {"error": f"upload_failed: {type(exc).__name__}"[:200]}

    if resp.status_code != 200:
        # Текст ответа приложения полезен в логе прогона: 403 подписи и 413
        # размера — разные аварии, и различать их надо на этой стороне тоже.
        return {"error": f"upload_http_{resp.status_code}: {resp.text[:150]}"}

    return {"stored": True, "bytes": len(mp4), "sha256": digest}


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
        size = int(inp.get("size", 512))
    except (TypeError, ValueError):
        return {"error": "bad_params"}

    if not (1 <= seconds <= MAX_SECONDS):
        return {"error": "bad_seconds"}
    if not (8 <= fps <= 15) or size not in (512, 640) or not (4 <= shot_seconds <= 15):
        return {"error": "bad_params"}

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

    t2i, i2i = _load()
    t_first = t_loop = 0.0
    ref = None
    per_frame = []
    shots = 0

    t0 = time.time()
    enc = _Encoder(size, size, fps, mark)
    try:
        for i in range(total):
            # Начало шота: композиция ставится ЗАНОВО. Без этого зум за минуты
            # уводит камеру внутрь детали — см. README, пункт 1.
            if i in shot_starts:
                shots += 1
                g = torch.Generator("cuda").manual_seed(seed + shots * 7919)
                ts = time.time()
                img = t2i(prompt=frame_prompts[i], negative_prompt=NEG,
                          num_inference_steps=4,
                          guidance_scale=1.5, width=size, height=size,
                          generator=g).images[0]
                t_first += time.time() - ts
                cur = np.array(img)
                ref = _lab_stats(cur)          # цветовой якорь СВОЙ у каждого шота
                enc.write(cur)
                continue

            ts = time.time()
            moved = _warp(cur, mo["zoom"], mo["angle"], mo["tx"], mo["ty"])
            out = i2i(prompt=frame_prompts[i], negative_prompt=NEG,
                      image=Image.fromarray(moved), strength=mo["strength"],
                      num_inference_steps=steps, guidance_scale=1.5,
                      generator=g).images[0]
            cur = _color_match(np.array(out), ref)
            enc.write(cur)
            dt = time.time() - ts
            t_loop += dt
            per_frame.append(dt)

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
        "width": size, "height": size, "fps": fps,
        "n_frames": total, "seconds": round(total / fps, 2), "shots": shots,
        "gen_seconds": round(gen, 1),
        "sec_per_frame": round(float(np.median(per_frame)), 3) if per_frame else None,
        "steps_per_frame": steps,
        "timings": {"first_frames": round(t_first, 1), "loop": round(t_loop, 1),
                    "encode": round(t_encode, 1)},
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
