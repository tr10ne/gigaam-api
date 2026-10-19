# gigaam-api

HTTP service that turns a **stereo** call recording into a speaker-separated transcript with
emotion scores. Built on [GigaAM](https://github.com/salute-developers/GigaAM) (ASR and
emotion models) and [Silero VAD](https://github.com/snakers4/silero-vad) for speech
segmentation.

The stereo assumption replaces heavy diarisation: the left channel is treated as the
client, the right channel as the operator. Each channel is sliced by VAD, every speech
fragment is transcribed, and emotion probabilities are estimated per fragment.

## API

### `POST /analyze`

`multipart/form-data` with `audio_file`; optional query parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `min_silence_duration_ms` | `500` | pause that splits a phrase |
| `speech_pad_ms` | `100` | padding around detected speech |

Returns a list of utterances sorted by start time:

```json
[
  {
    "text": "здравствуйте, меня зовут анна",
    "start": 1.24,
    "end": 3.9,
    "speaker": "OPERATOR",
    "emotions": {"neutral": 0.81, "positive": 0.12, "...": 0.0},
    "dominant_emotion": "neutral"
  }
]
```

Mono input is rejected with 400: without two channels the speaker labels would be a guess.

## Behaviour and limits

- audio is resampled to 16 kHz and processed per channel;
- fragments longer than 25 seconds are split, because GigaAM accepts up to 25 seconds;
- `torch.load` is patched for PyTorch 2.6+, where the default `weights_only=True` breaks
  the model checkpoints;
- models load once at startup and stay in memory — the first request is the slowest;
- CPU works, CUDA is used automatically when available (`GIGAAM_DEVICE`).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GIGAAM_MODEL` | `v3_e2e_rnnt` | ASR model; `v3_e2e_rnnt` is the phone-call default |
| `GIGAAM_DEVICE` | `cuda` if available, else `cpu` | inference device |

## Run

```bash
docker build -t gigaam-api .
docker run --rm -p 9000:9000 gigaam-api
```

```bash
curl -F "audio_file=@call.wav" \
  "http://127.0.0.1:9000/analyze?min_silence_duration_ms=500&speech_pad_ms=100"
```

Model checkpoints are downloaded on first start; mount a cache volume at `/root/.cache` to
keep them between runs.

## License

MIT for this service. GigaAM models and code are MIT (salute-developers), Silero VAD is
MIT as well.
