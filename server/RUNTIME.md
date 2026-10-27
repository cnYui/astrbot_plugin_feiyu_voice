# GPU runtime setup

For new installations, follow [INSTALL_AGENT.md](INSTALL_AGENT.md). It explains GPU detection, choosing CUDA, WSL ROCm or native-Windows/Linux Intel XPU, isolated per-engine environments, model downloads, generated profiles, and actual HTTP/audio acceptance. AMD profiles are experimental and have not been tested on physical AMD hardware. Linux XPU has passed all engine diagnostics plus TTS/RVC HTTP audio validation on one Arc A770M; singing audio remains pending.

`runtime_profiles/cuda.json`, `runtime_profiles/rocm.json` and `runtime_profiles/xpu.json` pin the selected torch/torchaudio builds. `scripts/setup_runtime.py` installs a new environment; `scripts/register_runtime.py` validates and registers an existing compatible one. Neither changes model weights. A service entry's optional `runtime_profile` selects the new adapter; omission preserves the legacy package-local runner.

RVC uses the pinned Applio source plus `runtime_compat/Applio/` and the SHA-verified upstream ContentVec/RMVPE assets listed in `runtime_compat/upstream_assets.json`. TTS keeps the model checkpoint/reference selection, uses standard PyTorch inference and explicitly selects CPU ONNX sessions. SVS uses the fixed SoulX-Singer commit and configurable precision. Training and optional CUDA-only acceleration dependencies are excluded from the inference requirements.

Legacy deployments may keep their existing package `runtime.local.json`. The original TTS package runner's CUDA version pin does not apply to the independent managed adapter, which checks its own generated profile plus the unchanged package and source hashes. Never bypass or edit the model package's integrity files to change a GPU backend.

Versioned source commits, resolved dependency snapshots, source hashes, kernel diagnostics and audio receipts serve different purposes. A registered profile or successful diagnostic does not certify model audio quality or support for every GPU in a series.
