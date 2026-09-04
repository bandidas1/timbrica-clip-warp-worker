"""Локальный прогон handler.py без RunPod и без Docker.

Зачем: проверить главное проектное решение — что склейки каждые shot_seconds
РЕАЛЬНО не дают композиции уползти. Без склеек тест «портрет» к 130-му кадру
оказывался внутри глаза (замер 2026-09-03).

Запуск (веса тянутся из кэша HuggingFace, а не из /models):
    MODEL_DIR=Lykon/dreamshaper-8 LORA_DIR=latent-consistency/lcm-lora-sdv1-5 \
    python local_smoke.py
"""
import base64
import os
import sys
import types

# Заглушка runpod: handler.py на импорте регистрирует serverless-хендлер.
stub = types.ModuleType("runpod")
stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
sys.modules["runpod"] = stub

os.environ.setdefault("MODEL_DIR", "Lykon/dreamshaper-8")
os.environ.setdefault("LORA_DIR", "latent-consistency/lcm-lora-sdv1-5")

import handler as H  # noqa: E402


# Локальный прогон берёт модель из кэша HF, где лежат ОБЫЧНЫЕ имена файлов, а не
# fp16-варианты, запечённые в образ. Без этого загрузчик искал бы `*.fp16.*` и
# скачивал их заново. Значение по умолчанию в handler.py рассчитано на образ.
os.environ.setdefault("MODEL_VARIANT", "")


def _rss_mb():
    """МГНОВЕННЫЙ RSS процесса, не пик — назвать это «пиком» было бы врать:
    замер берётся один раз до и один раз после, всплеск между ними невидим.
    Для проверки потокового кодирования этого достаточно (кадры копились бы
    монотонно), но для охоты за всплесками нужен опрос в отдельном потоке."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return None


def _args():
    p = argparse.ArgumentParser(description="Локальный прогон handler.py")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--out", default=None, help="апскейл выхода, например 720x1280")
    p.add_argument("--seconds", type=float, default=float(os.environ.get("SMOKE_SECONDS", 15)))
    p.add_argument("--shot", type=float, default=float(os.environ.get("SMOKE_SHOT", 5)))
    p.add_argument("--image", default=None, help="стартовая картинка (обложка) — путь к файлу")
    return p.parse_args()


def main():
    a = _args()
    seconds, shot = a.seconds, a.shot
    rss_before = _rss_mb()

    inp = {
        # Портрет — самый злой случай для сноса композиции.
        "prompt": ("close-up portrait of a young woman, freckles, soft rim light, "
                   "shallow depth of field, photorealistic, 85mm lens"),
        "style": "push",          # самый агрессивный зум из наборов
        "seconds": seconds,
        "fps": 12,
        "shot_seconds": shot,
        "width": a.width,
        "height": a.height,
        "seed": 20260903,
    }
    if a.out:
        w, h = a.out.lower().split("x")
        inp["out"] = [int(w), int(h)]
    if a.image:
        with open(a.image, "rb") as fh:
            inp["init_image_b64"] = base64.b64encode(fh.read()).decode()

    res = H.handler({"input": inp})

    if "error" in res:
        print("ОШИБКА:", res["error"])
        return 1

    mp4 = base64.b64decode(res.pop("mp4_b64"))
    out = os.path.join(os.path.dirname(__file__), "smoke.mp4")
    with open(out, "wb") as f:
        f.write(mp4)

    print(f"→ {out}  ({len(mp4)/1024/1024:.2f} МБ)")
    for k, v in res.items():
        print(f"   {k}: {v}")

    rss_after = _rss_mb()
    if rss_before and rss_after:
        # Копящий вариант рос бы на ширина×высота×3 байта за кадр.
        would_be = res["n_frames"] * a.width * a.height * 3 / 1024 / 1024
        print(f"   память: {rss_before} → {rss_after} МБ (+{rss_after - rss_before}); "
              f"копящий вариант держал бы ещё ~{would_be:.0f} МБ кадров")

    expected_shots = max(1, int(seconds / shot))
    if res["shots"] < expected_shots:
        print(f"⚠️ склеек {res['shots']}, ожидалось ≥{expected_shots} — "
              f"перестановка композиции НЕ работает")
        return 1
    print(f"✅ склеек {res['shots']} — композиция переставляется")
    return 0


if __name__ == "__main__":
    sys.exit(main())
