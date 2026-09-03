"""Запекает веса в образ на этапе сборки: холодный старт = pull образа, не скачивание.

Берём ТОЛЬКО fp16-safetensors и только нужные подпапки — полный репозиторий
dreamshaper-8 весит 5.1 ГБ из-за дублей в разных форматах, а нужного там ~2 ГБ.
"""
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

    total = 0
    for d in (sd_dir, lora_dir):
        for root, _, files in os.walk(d):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print(f"веса запечены: {total / 1024 ** 3:.2f} ГБ")


if __name__ == "__main__":
    main()
