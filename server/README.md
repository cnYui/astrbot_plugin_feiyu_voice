# Feiyu Voice Service v0.2.0

For AI agent installation, start with [INSTALL_AGENT.md](INSTALL_AGENT.md): select NVIDIA CUDA, AMD WSL ROCm or Intel XPU (native Windows/Linux), install independent inference environments, download the models, and select generated `runtime_profile` files. The API uses a separate CPU environment. Choose `configs/voice_service.cuda.example.json`, `configs/voice_service.rocm.example.json` or `configs/voice_service.xpu.example.json`, then save your machine's settings as `configs/voice_service.json`. AMD is experimental without physical GPU validation. Legacy package-local runtime configuration remains supported.

```bash
python -m venv .venv
# Activate .venv using your platform's standard command
python -m pip install -r requirements-service.txt
python -m feiyu_voice_service --host 127.0.0.1 --port 9880
```

The private API key is created in `.runtime/voice-service/api_key.txt`. Copy it into AstrBot's plugin configuration. Never publish it. Open `/docs` for the authenticated API schema.

Endpoints: GET `/health`, GET `/v1/capabilities`, POST `/v1/assets`, POST `/v1/speech`, POST `/v1/voice-conversion`, POST `/v1/singing`, GET `/v1/jobs/{id}`, GET `/v1/jobs/{id}/audio`, POST `/v1/admin/reload`. Private routes require `Authorization: Bearer <key>`.

Speech JSON: `{"voice":"feiyu","text":"你好呀","style":"cute"}`. Conversion JSON: `{"voice":"feiyu","audio_id":"<uploaded asset id>"}`. Singing JSON: `{"voice":"feiyu","score_id":"<uploaded asset id>","lyrics":"微光悠_悠"}`. Upload WAV/FLAC, MIDI/ABC, TXT or region JSON using multipart field `file`.

Submission returns HTTP202 and a job ID. Poll until `success`, then download WAV. One worker serializes GPU jobs; do not run multiple Uvicorn workers. Existing unfinished jobs become `interrupted` after a restart. Uploads, requests and outputs remain private in `.runtime/`; manage retention locally. Configured package replacement is supported while the queue is idle via `/v1/admin/reload`; a different engine needs its own adapter. Current public RVC engine is `feiyu_rvc_release`.

Ordinary TTS uses its own model directly. Voice conversion uses RVC. Score singing needs the optional SVS setup and then uses the selected RVC. Manual breath processing is never automatic. MIDI/ABC lyrics currently support Mandarin, one character per note, `_` for melisma; expand ABC repeats before importing.

Validation scope and results: [VALIDATION.json](VALIDATION.json). AMD hardware testing remains pending. Intel XPU is validated on one Windows Arc 130T iGPU and one Linux Arc A770M; Linux singing audio remains pending.
