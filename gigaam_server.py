import os
import tempfile
import gigaam
import torch
torch.backends.nnpack.enabled = False 

import torchaudio
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

# --- ФИКС ДЛЯ PyTorch 2.6+ ---
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, "weights_only": False})
# --------------------------------

app = FastAPI(title="GigaAM Call Center API (Stereo + Silero VAD)")

# Для телефонии v3_rnnt работает лучше всего
ASR_MODEL_NAME = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
DEVICE         = os.getenv("GIGAAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

asr_model = None
emo_model = None
vad_model = None
get_speech_timestamps = None

@app.on_event("startup")
async def load_models():
    global asr_model, emo_model, vad_model, get_speech_timestamps
    
    print(f"Loading ASR model ({ASR_MODEL_NAME}) on {DEVICE}...")
    asr_model = gigaam.load_model(ASR_MODEL_NAME, device=DEVICE)
    asr_model.eval()
    
    print(f"Loading Emotion model on {DEVICE}...")
    emo_model = gigaam.load_model("emo", device=DEVICE)
    emo_model.eval()
    
    print(f"Loading Silero VAD on {DEVICE}...")
    from silero_vad import load_silero_vad, get_speech_timestamps as get_ts
    vad_model = load_silero_vad(onnx=False).to(DEVICE)
    get_speech_timestamps = get_ts
    
    print("Models loaded successfully! Heavy Pyannote Diarization is DISABLED (using Stereo separation).")

def get_emotions(waveform: torch.Tensor):
    
    # Убеждаемся, что у тензора есть размерность каналов [1, Time]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        
    try:
        # Сохраняем кусок аудио во временный файл
        torchaudio.save(tmp_path, waveform.cpu(), 16000)
        
        # Передаем ПУТЬ К ФАЙЛУ в модель (как она и требует)
        probs = emo_model.get_probs(tmp_path)
        
        if isinstance(probs, list):
            probs = probs[0]
            
        emotions = {str(k): round(float(v), 4) for k, v in probs.items()}
        dominant = max(emotions, key=emotions.get)
        return emotions, dominant
    finally:
        # Обязательно удаляем временный файл
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def process_channel(
    channel_waveform: torch.Tensor, 
    speaker_name: str, 
    min_silence_duration_ms: int,
    speech_pad_ms: int
):
    """
    Нарезает моно-канал с помощью Silero VAD и распознает каждый кусочек.
    """
    # 1. Получаем таймстемпы речи (только голос, игнорируем шум АТС)
    speech_timestamps = get_speech_timestamps(
        channel_waveform,
        vad_model,
        threshold=0.4, # Оптимальный порог уверенности
        min_speech_duration_ms=300, # Игнорируем короткие вздохи
        min_silence_duration_ms=min_silence_duration_ms, # Пауза для разделения фраз
        speech_pad_ms=speech_pad_ms, # "Подушка" вокруг речи, чтобы не съедать первые буквы
        sampling_rate=16000,
        return_seconds=True
    )
    
    utterances = []
    
    for ts in speech_timestamps:
        start_sec = ts['start']
        end_sec = ts['end']
        
        # Вырезаем нужный кусок аудио по семплам (1 sec = 16000 samples)
        start_sample = int(start_sec * 16000)
        end_sample = int(end_sec * 16000)
        chunk_waveform = channel_waveform[start_sample:end_sample].to(DEVICE)
        
        duration_sec = end_sec - start_sec
        
        MAX_CHUNK_DUR = 25.0 # Лимит GigaAM
        
        if duration_sec > MAX_CHUNK_DUR:
            # Бьем длинный кусок на под-куски по 25 секунд
            sub_texts = []
            num_subchunks = int(torch.ceil(torch.tensor(duration_sec / MAX_CHUNK_DUR)).item())
            
            for i in range(num_subchunks):
                sub_start = i * int(MAX_CHUNK_DUR * 16000)
                sub_end = min((i + 1) * int(MAX_CHUNK_DUR * 16000), chunk_waveform.shape[0])
                
                sub_waveform = chunk_waveform[sub_start:sub_end]
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                try:
                    torchaudio.save(tmp_path, sub_waveform.cpu().unsqueeze(0), 16000)
                    sub_text = asr_model.transcribe(tmp_path)
                    if sub_text:
                        sub_texts.append(sub_text.strip())
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            text = " ".join(sub_texts)
            
        else:
            # Обычная обработка куска < 25 сек
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                torchaudio.save(tmp_path, chunk_waveform.cpu().unsqueeze(0), 16000)
                text = asr_model.transcribe(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        # -------------------------------------------------------------
        text = text.strip() if text else ""
        
        if not text:
            continue
            
        # Оценка эмоций для всего куска
        emotions, dominant = get_emotions(chunk_waveform)
        
        utterances.append({
            "text": text,
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "speaker": speaker_name,
            "emotions": emotions,
            "dominant_emotion": dominant
        })
    
    return utterances 

@app.post("/analyze")
async def analyze_stereo_call(
    audio_file: UploadFile = File(...),
    # Параметры нарезки VAD
    min_silence_duration_ms: int = Query(500, description="Пауза для разрыва фразы, мс. Рекомендуется 500 для e2e моделей"),
    speech_pad_ms: int = Query(100, description="Отступ по краям речи, мс")
):
    ext = os.path.splitext(audio_file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await audio_file.read())
        orig_path = tmp.name

    try:
        # Загружаем аудио через torchaudio (намного эффективнее pydub для нейросетей)
        # Для чтения mp3 может потребоваться установленный ffmpeg (или соберите docker с ним)
        waveform, sample_rate = torchaudio.load(orig_path)
        
        # Ресемплинг к 16000Hz (требование GigaAM и VAD)
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)
        
        # Защита от моно-файлов
        if waveform.shape[0] < 2:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Audio file must be stereo (2 channels) for exact speaker separation")
            
        # Разделяем на каналы (Torchaudio возвращает тензор [channels, time])
        left_channel  = waveform[0] # Обычно клиент
        right_channel = waveform[1] # Обычно оператор
        
        # Обрабатываем независимо
        client_utterances   = process_channel(left_channel, "CLIENT", min_silence_duration_ms, speech_pad_ms)
        operator_utterances = process_channel(right_channel, "OPERATOR", min_silence_duration_ms, speech_pad_ms)
        
        # Объединяем и сортируем
        all_utterances = client_utterances + operator_utterances
        all_utterances.sort(key=lambda x: x["start"])
        
        return JSONResponse(all_utterances)

    finally:
        if os.path.exists(orig_path):
            os.unlink(orig_path)