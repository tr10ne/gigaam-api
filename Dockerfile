FROM python:3.10-slim

# Устанавливаем системные зависимости (ffmpeg обязателен для обработки аудио)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    git \
    build-essential \
    libsndfile1 \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Обновляем pip
RUN pip install --upgrade pip

# 2. Устанавливаем torch
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Ставим остальные пакеты
RUN pip install --no-cache-dir \
    "git+https://github.com/salute-developers/GigaAM.git" \
    silero-vad \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile \
    sox

# Копируем код сервера
COPY gigaam_server.py ./

EXPOSE 9000

# Запускаем сервер
CMD ["uvicorn", "gigaam_server:app", "--host", "0.0.0.0", "--port", "9000"]