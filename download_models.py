"""Запекает веса в образ на этапе сборки: холодный старт = pull образа, не скачивание.

Берём ТОЛЬКО fp16-safetensors и только нужные подпапки — полный репозиторий
dreamshaper-8 весит 5.1 ГБ из-за дублей в разных форматах, а нужного там ~2 ГБ.
"""
import json
import os
from huggingface_hub import snapshot_download

# Обе лицензии — OpenRAIL-M, коммерческое использование разрешено.
# ⛔ Не подменять на SDXL-Turbo / FLUX.1-dev / flux.1-lite: non-commercial.
SD15 = "Lykon/dreamshaper-8"
LCM = "latent-consistency/lcm-lora-sdv1-5"


def main():
    sd_dir = os.environ.get("MODEL_DIR", "/models/sd15")
    lora_dir = os.environ.get("LORA_DIR", "/models/lcm-lora-sdv1-5")

    snapshot_download(
        SD15, local_dir=sd_dir,
        allow_patterns=[
            "model_index.json",
            "scheduler/*", "tokenizer/*",
            # ⚠️ feature_extractor нужен, даже когда safety_checker выключен:
            # он объявлен в model_index.json, и загрузчик всё равно его ищет.
            # Без него падение приходит ПОСЛЕ unet-весов, то есть следующим
            # холодным стартом — поэтому список компонентов теперь сверяется
            # с model_index.json, а не с памятью автора (см. _verify).
            "feature_extractor/*",
            "text_encoder/*.json", "text_encoder/*fp16.safetensors",
            "unet/*.json", "unet/*fp16.safetensors",
            "vae/*.json", "vae/*fp16.safetensors",
        ],
        # .bin и .ckpt — те же веса в других форматах, чистый вес образа.
        ignore_patterns=["*.bin", "*.ckpt", "*.msgpack", "*.onnx", "*.png", "*.jpg"],
    )

    snapshot_download(LCM, local_dir=lora_dir,
                      allow_patterns=["*.safetensors", "*.json"],
                      ignore_patterns=["*.bin"])

    _verify(sd_dir, lora_dir)


def _verify(sd_dir, lora_dir):
    """Сборка НЕ ИМЕЕТ ПРАВА выпустить образ без весов.

    ⚠️⚠️ Здесь раньше стоял просто print с итоговым размером — и он честно
    печатал маленькое число, когда шаблон имён не совпал с репозиторием. Сборка
    при этом проходила, образ уезжал в реестр, а падало всё на ЖИВОМ прогоне
    через 19.6 минуты холодного старта: «no file named diffusion_pytorch_model.bin
    found in /models/sd15/unet». Напечатанное свидетельство, которое никто не
    проверяет, — не проверка. Теперь несоответствие роняет сборку.
    """
    problems = []

    # Список компонентов берём из model_index.json, а НЕ из головы: пайплайн
    # грузит именно его, и любая забытая папка (feature_extractor в первой
    # версии) роняет прогон уже в образе. Так проверка не может отстать от
    # модели — она задаёт вопрос тому же источнику, что и загрузчик.
    index_path = os.path.join(sd_dir, "model_index.json")
    if not os.path.isfile(index_path):
        raise SystemExit("запекание весов НЕ УДАЛОСЬ: нет model_index.json")
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)

    # Веса ждём только там, где они бывают; у токенизатора и планировщика их нет.
    heavy = {"unet": 1_000_000_000, "vae": 100_000_000, "text_encoder": 150_000_000}

    for name, spec in index.items():
        if name.startswith("_") or not isinstance(spec, list):
            continue
        if spec[0] is None:          # компонент отключён в самой модели
            continue
        if name == "safety_checker":  # мы его выключаем и не запекаем намеренно
            continue
        d = os.path.join(sd_dir, name)
        if not os.path.isdir(d) or not os.listdir(d):
            problems.append(f"{name}: папки нет или она пуста, а model_index.json её требует")
            continue
        if name in heavy:
            weights = [f for f in os.listdir(d) if f.endswith((".safetensors", ".bin"))]
            size = sum(os.path.getsize(os.path.join(d, f)) for f in weights)
            if not weights:
                problems.append(f"{name}: файлов весов НЕТ (есть только конфиги)")
            elif size < heavy[name]:
                problems.append(f"{name}: {size / 1024 ** 2:.0f} МБ — меньше ожидаемого")

    lora = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")] \
        if os.path.isdir(lora_dir) else []
    if not lora:
        problems.append("lcm-lora: файла весов НЕТ")

    total = 0
    for d in (sd_dir, lora_dir):
        for root, _, files in os.walk(d):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)

    if problems:
        raise SystemExit(
            "запекание весов НЕ УДАЛОСЬ — образ собирать нельзя:\n  "
            + "\n  ".join(problems)
            + f"\n(итого {total / 1024 ** 3:.2f} ГБ; проверь allow_patterns и то, "
              "какие имена файлов есть в репозитории модели)"
        )

    print(f"веса запечены и проверены: {total / 1024 ** 3:.2f} ГБ")


if __name__ == "__main__":
    main()
