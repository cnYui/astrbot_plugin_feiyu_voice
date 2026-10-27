"""CosyVoice3 adapter preserving release weights and explicit CPU ONNX sessions."""
import argparse
import os
from pathlib import Path
import sys
import time

from feiyu_runtime.common import (apply_backend_env, checked_path, cpu_onnx_sessions, device_type, load,
                                 prepare, save, sha, verify_audio_library, verify_package, verify_repo)


def move_to_xpu(torch, voice, device):
    device = torch.device(device)
    for name in ("llm", "flow", "hift"):
        getattr(voice.model, name).to(device).eval()
    voice.model.device = device
    voice.frontend.device = device
    if sys.platform == "linux":
        # Linux XPU cannot run the float64 weight_norm kernel used by this predictor.
        f0 = voice.model.hift.f0_predictor
        f0.to("cpu").eval()
        original_forward = f0.forward

        def f0_forward(speech_feat, *args, **kwargs):
            return original_forward(speech_feat.cpu(), *args, **kwargs)

        f0.forward = f0_forward


def main(package, profile, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = verify_package(package)
    if manifest["schema"] != "feiyu-voice-local-release-v1":
        raise ValueError("Unsupported TTS package")
    repo = verify_repo(profile)
    request = load(args.request)
    selected = manifest["styles"][request.get("style", manifest["default_style"])]
    out = Path(request["output_dir"]).resolve()
    if out.is_relative_to(package) or out == repo or repo.is_relative_to(out):
        raise ValueError("Output must be outside the package and runtime source")
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
    apply_backend_env(profile)
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "third_party/Matcha-TTS")]
    import numpy as np
    import soundfile as sf
    import torch
    import onnxruntime as ort
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed

    info = prepare(torch, profile)
    verify_audio_library(profile)
    sessions = []
    started = time.perf_counter()
    with cpu_onnx_sessions(ort, sessions):
        voice = AutoModel(model_dir=str(checked_path(package, manifest["base_model_dir"])),
                          fp16=False, load_trt=False, load_vllm=False)
    checkpoint = checked_path(package, selected["llm_checkpoint"])
    reference = checked_path(package, selected["reference_audio"])
    if sha(checkpoint) != selected["llm_sha256"] or sha(reference) != selected["reference_sha256"]:
        raise ValueError("Selected TTS assets differ from manifest")
    if checkpoint != checked_path(package, manifest["base_model_dir"]) / "llm.pt":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        voice.model.llm.load_state_dict(state, strict=True)
        del state
    if info["backend"] == "xpu":
        # Upstream only knows cuda/CPU: it loads on CPU, then the networks and the
        # device used for its input tensors are moved to the Intel GPU.
        move_to_xpu(torch, voice, info["device"])
    # Verify every main network's actual device.
    for name in ("llm", "flow", "hift"):
        actual = next(getattr(voice.model, name).parameters()).device.type
        if actual != device_type(info["backend"]):
            raise RuntimeError("Unexpected TTS network device: " + name)
    voice.model.llm.eval()
    params = manifest["inference"]
    set_all_random_seed(params["seed"])
    save(out / "job.json", dict(checkpoint_sha256=sha(checkpoint), reference_sha256=sha(reference),
                              manifest_sha256=sha(package / "manifest.json"), inference=params, runtime=info))
    with torch.inference_mode():
        chunks = [x["tts_speech"].detach().cpu().numpy()[0] for x in voice.inference_zero_shot(
            request["text"], params["prompt_prefix"] + selected["reference_text"], str(reference),
            stream=False, speed=params["speed"], text_frontend=True)]
    if not chunks:
        raise ValueError("TTS returned no audio")
    audio = np.concatenate(chunks)
    if len(audio) <= 2400 or not np.isfinite(audio).all() or abs(audio).max() <= 1e-4:
        raise ValueError("Invalid TTS output")
    sf.write(out / "native.wav", audio, voice.sample_rate, subtype="FLOAT")
    gain = min(1., .98 / float(abs(audio).max()))
    sf.write(out / "speech.wav", audio * gain, voice.sample_rate, subtype="PCM_16")
    save(out / "result.json", dict(status="success", playback_sha256=sha(out / "speech.wav"),
        output_sha256=sha(out / "native.wav"), sample_rate=voice.sample_rate, seconds=len(audio) / voice.sample_rate,
        playback_gain=gain, runtime=info, onnx_sessions=sessions, actual_text_frontend=voice.frontend.text_frontend,
        checkpoint_sha256=sha(checkpoint), elapsed_seconds=time.perf_counter() - started, runtime_rvc=False))
    return 0
