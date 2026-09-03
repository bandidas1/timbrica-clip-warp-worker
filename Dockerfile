# Генератор видеоклипа для RunPod serverless — warp-петля SD 1.5 + LCM.
#
# Движок выбран замером, а не вкусом: из четырёх подходов только этот прошёл все
# четыре тестовых сценария без вырождения (0.93 с/кадр на RTX 4060 8 ГБ).
# Полный SDXL и FLUX-класс отвергнуты, подробности — README.md.
#
# Лицензии: dreamshaper-8 и lcm-lora-sdv1-5 — OpenRAIL-M, коммерчески пригодны.
# ⛔ SDXL-Turbo / FLUX.1-dev / flux.1-lite здесь появиться НЕ МОГУТ: non-commercial.
#
# Веса (~2 ГБ) запекаются в образ: холодный старт = pull образа и загрузка
# модулей, а не скачивание чекпойнта.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models/sd15 \
    LORA_DIR=/models/lcm-lora-sdv1-5 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# torch первым и своим индексом — держим версию под CUDA 12.4 и не даём
# резолверу requirements.txt утащить её вниз.
#
# ⚠️⚠️ 2.6.0, а НЕ 2.4.1 (как было до замера на живой карте 03.09). transformers
# 5.x требует torch >= 2.5 и при более старом МОЛЧА отключает интеграцию с
# torch — а дальше падает не он, а diffusers, с `name 'nn' is not defined`
# внутри загрузчика LoRA. Ошибка не называет ни настоящего виновника, ни версии,
# и локально не воспроизводится вовсе, если там torch новее. Пара
# torch↔transformers здесь связана жёстко: поднимая одну, проверяй другую.
#
# ⚠️ torchvision/torchaudio НЕ ставим намеренно: handler.py их не использует, а
# transformers подхватывает их только если они есть. На стендовом поде RunPod
# (образ runpod/pytorch несёт согласованную тройку) это дало каскад — сначала
# `operator torchvision::nms does not exist`, потом undefined symbol в
# libtorchaudio: поднятая половина набора ломается тише, чем не поднятая.
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

# Веса — отдельным слоем ДО кода: правка handler.py не пересобирает 2 ГБ.
COPY download_models.py .
RUN python download_models.py

COPY handler.py .

CMD ["python", "-u", "handler.py"]
