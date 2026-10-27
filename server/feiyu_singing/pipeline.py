"""Local orchestration; independent SVS/RVC processes release GPU memory between stages."""
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


_native_host = False


def set_native_host(runtime):
    """Profiles with host=windows run natively on Windows; others keep the WSL path mapping."""
    global _native_host
    _native_host = os.name == "nt" and runtime.get("host") == "windows"


def linux_path(path):
    path = Path(path).resolve()
    if os.name != "nt" or _native_host:
        return str(path)
    value = path.as_posix()
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("WSL 推理需要本地盘符路径，不支持 UNC 网络路径。")
    return "/mnt/" + value[0].lower() + value[2:]


def runtime_command(runtime, script, *arguments):
    set_native_host(runtime)
    command = [runtime["python"], "-B", linux_path(script), *map(str, arguments)]
    if runtime.get("python_library_dir") and not _native_host:
        command[:0] = ["env", "LD_LIBRARY_PATH=" + runtime["python_library_dir"]]
    if os.name == "nt" and not _native_host:
        command = ["wsl.exe", "-d", runtime["distribution"], "-e", *command]
    return command


def run_logged(command, log_path):
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        last_message = time.monotonic()
        while process.poll() is None:
            time.sleep(.5)
            if time.monotonic() - last_message > 25:
                print(f"仍在运行，日志：{log_path}", flush=True)
                last_message = time.monotonic()
        if process.returncode:
            tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-3500:]
            raise RuntimeError(f"推理进程退出码 {process.returncode}，详见 {log_path}\n{tail}")


def input_copy(source, folder, name):
    source = Path(source).resolve()
    destination = folder / name
    shutil.copyfile(source, destination)
    return dict(original=str(source), snapshot=linux_path(destination), sha256=sha(destination))


def finalize_audio(source, destination, expected_seconds):
    """Keep native RVC intact. Only pad a short missing tail to the score length."""
    import numpy as np
    import soundfile as sf
    wave, rate = sf.read(source, dtype="float64")
    info = sf.info(source)
    if rate != 40000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("最终 RVC 音频必须是有限数值的 40kHz 单声道人声。")
    if np.sqrt(np.mean(wave ** 2)) < 1e-7:
        raise ValueError("最终 RVC 音频是静音。")
    expected = round(expected_seconds * rate)
    missing = expected - len(wave)
    if not 0 <= missing <= round(.1 * rate):
        raise ValueError(f"RVC 输出与谱面时长不匹配：{len(wave) / rate:.3f}s / {expected_seconds:.3f}s。原生输出已保留。")
    if missing:
        sf.write(destination, np.pad(wave, (0, missing)), rate, subtype=info.subtype)
    else:
        shutil.copyfile(source, destination)
    delivered, delivered_rate = sf.read(destination, dtype="float64")
    if delivered_rate != rate or not np.array_equal(wave, delivered[:len(wave)]):
        raise ValueError("输出校验失败：补尾静音前的音频样本发生变化。")
    return dict(sample_rate=rate, channels=1, encoding=info.subtype, frames=len(delivered),
                duration_seconds=len(delivered) / rate, native_duration_seconds=len(wave) / rate,
                tail_silence_padding_samples=missing, original_samples_unchanged=True,
                finite=True, peak=float(abs(delivered).max()),
                near_full_scale_samples=int((abs(delivered) >= .999).sum()))


def make_preview(folder, compiled, info):
    """Local HTML with exact alignment and playable intermediate/final vocals."""
    rows = []
    for i, event in enumerate(compiled["events"]):
        if not event["pitch"]:
            continue
        values = [i + 1, event["text"], event["phoneme"], event["pitch"],
                  f'{event["start_frame"] / 50:.2f}', f'{event["end_frame"] / 50:.2f}',
                  "延音" if event["note_type"] == 3 else "新字"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    player = ""
    for name, label in [("singing.wav", "肥鱼清唱"), ("svs/generated.wav", "SVS 中间歌声")]:
        if (folder / name).is_file():
            player += f'<p>{label}</p><audio controls preload="metadata" src="{name}"></audio>'
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>肥鱼按谱唱歌</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:36px auto;padding:0 20px;background:#f6f9fb;color:#203448}}audio{{width:100%}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{text-align:left;border-bottom:1px solid #dae3e9;padding:9px}}h1{{font-size:26px}}</style>
<h1>肥鱼按谱唱歌 · 本地试听与对齐</h1><p>换声包：{html.escape(info['rvc_package'])} · 时长 {compiled['duration_seconds']:.2f} 秒 · 移调 {compiled['transpose_semitones']} 半音</p>
<p>拼音：{html.escape(compiled['pinyin_source'])}。D76：{html.escape(info.get('D76_status', '尚未生成'))}。音质和咬字待试听确认。</p>{player}
<p><a href="alignment.json">完整对齐 JSON</a> · <a href="target_metadata.json">模型谱面 JSON</a> · <a href="result.json">任务结果</a></p>
<table><thead><tr><th>事件</th><th>唱字</th><th>读音</th><th>MIDI 音高</th><th>开始/秒</th><th>结束/秒</th><th>类型</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>'''
    (folder / "试听与对齐.html").write_text(text, encoding="utf-8")


def synthesize(score, *, lyrics=None, lyrics_file=None, pinyin_file=None, track=None,
               channel=None, bpm=None, midi_encoding="utf-8", transpose=0, segment_seconds=20, output_dir=None,
               config=None, prepare_only=False, regions=None, seed=42):
    from .score import compile_score, read_score
    if lyrics is not None and lyrics_file:
        raise ValueError("--lyrics 与 --lyrics-file 只能选一个。")
    score = Path(score).resolve()
    if lyrics_file:
        lyrics = Path(lyrics_file).read_text(encoding="utf-8-sig")
    pinyin = Path(pinyin_file).read_text(encoding="utf-8-sig") if pinyin_file else None
    notes, duration, score_info = read_score(score, track=track, channel=channel, bpm=bpm, midi_encoding=midi_encoding)
    compiled = compile_score(notes, duration, lyrics, pinyin, transpose, segment_seconds)
    cfg_path = resolve(config or "configs/singing.json")
    cfg = load(cfg_path)
    if cfg.get("svs_runtime_profile"):
        from feiyu_runtime.common import validate_profile
        profile_path = resolve(cfg["svs_runtime_profile"])
        if cfg.get("svs_runtime_profile_sha256") and sha(profile_path) != cfg["svs_runtime_profile_sha256"]:
            raise ValueError("SVS runtime profile changed after submission")
        cfg["svs_runtime"] = validate_profile(load(profile_path))
        set_native_host(cfg["svs_runtime"])
        if cfg["svs_runtime"]["engine"] != "svs":
            raise ValueError("SVS runtime profile engine mismatch")
    package = resolve(cfg["rvc_package"])
    manifest = load(package / "manifest.json")
    if manifest["id"] != cfg["rvc_package_id"]:
        raise ValueError("RVC 包身份与配置不一致。")
    phoneset = resolve(cfg["phoneset"]["path"])
    if sha(phoneset) != cfg["phoneset"]["sha256"]:
        raise ValueError("音素词表校验失败。")
    allowed = set(load(phoneset))
    unsupported = {e["phoneme"] for e in compiled["events"]} - allowed
    if unsupported:
        raise ValueError(f"模型不支持这些拼音：{sorted(unsupported)}。请用 --pinyin-file 指定受支持读音。")
    folder = Path(output_dir).resolve() if output_dir else ROOT / "outputs/singing" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    # Resolve symlinks and reject an existing destination before any write.
    folder = folder.resolve()
    protected = [score, cfg_path, phoneset, *[Path(p).resolve() for p in (lyrics_file, pinyin_file, regions) if p]]
    if folder.exists() or any(p.is_relative_to(folder) for p in protected):
        raise ValueError("输出目录已存在或包含输入文件，请指定全新的目录；不会覆盖既有结果。")
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(schema="feiyu-singing-result-v1", status="preparing", output_directory=str(folder),
                  rvc_package=manifest["id"], rvc_manifest_sha256=sha(package / "manifest.json"),
                  config_sha256=sha(cfg_path), score=score_info, duration_seconds=compiled["duration_seconds"],
                  source_type="score_to_svs_to_rvc", new_training_updates=0,
                  perceptual_review="pending", D76_status="未应用；须标记本曲换气区间")
    try:
        inputs = folder / "inputs"
        inputs.mkdir()
        result["inputs"] = dict(score=input_copy(score, inputs, "score" + score.suffix.lower()))
        (inputs / "lyrics.txt").write_text(lyrics if lyrics is not None else "\n".join(n.lyric for n in notes), encoding="utf-8")
        result["inputs"]["lyrics_sha256"] = sha(inputs / "lyrics.txt")
        if pinyin_file:
            result["inputs"]["pinyin"] = input_copy(pinyin_file, inputs, "pinyin.txt")
        if regions:
            result["inputs"]["regions"] = input_copy(regions, inputs, "regions.json")
        save(folder / "config_snapshot.json", cfg)
        save(folder / "alignment.json", {k: v for k, v in compiled.items() if k != "metadata"})
        save(folder / "target_metadata.json", compiled["metadata"])
        result["target_metadata_sha256"] = sha(folder / "target_metadata.json")
        if not 0 <= seed <= 2 ** 32 - 1:
            raise ValueError("seed 须在 0–4294967295 范围内。")
        if prepare_only:
            result["status"] = "prepared"
            save(folder / "result.json", result)
            make_preview(folder, compiled, result)
            print(f"谱面与歌词已校验：{folder}", flush=True)
            return result
        prompt = cfg["prompt"]
        prompt_items = {}
        for role in ("audio", "metadata"):
            item = prompt[role]
            path = resolve(item["path"])
            if sha(path) != item["sha256"]:
                raise ValueError(f"SVS 参考 {role} 与已验证版本不一致。")
            prompt_items[role] = dict(path=linux_path(path), sha256=item["sha256"])
        result["svs_prompt"] = dict(id=prompt["id"], annotation_status=prompt["annotation_status"], **prompt_items)
        svs = folder / "svs"
        svs.mkdir()
        job = dict(runtime=cfg["svs_runtime"], weight=dict(path=linux_path(resolve(cfg["svs_weight"]["path"])),
                                                        sha256=cfg["svs_weight"]["sha256"]),
                   prompt=prompt_items, phoneset=dict(path=linux_path(phoneset), sha256=sha(phoneset)),
                   target=dict(path=linux_path(folder / "target_metadata.json"), sha256=result["target_metadata_sha256"]),
                   output_dir=linux_path(svs), frames=compiled["frames"],
                   parameters=dict(control="score", auto_shift=False, pitch_shift=0,
                                   n_steps=32, cfg=1., fp16=cfg["svs_runtime"].get("precision", "float16") == "float16", seed=seed))
        save(svs / "job.json", job)
        result["status"] = "running_svs"
        save(folder / "result.json", result)
        print(f"正在按谱合成歌声（{len(compiled['metadata'])} 段）…", flush=True)
        run_logged(runtime_command(cfg["svs_runtime"], ROOT / "feiyu_singing/svs_worker.py",
                                   "--job", linux_path(svs / "job.json")), svs / "execution.log")
        sr = load(svs / "result.json")
        if sr["status"] != "success" or sha(svs / "generated.wav") != sr["output_sha256"]:
            raise ValueError("SVS 未产生有效结果。")
        result["svs"] = sr
        result["status"] = "running_rvc"
        save(folder / "result.json", result)
        print(f"正在使用 {manifest['id']} 转换音色…", flush=True)
        rvc = folder / "rvc"
        public_rvc = manifest["schema"] == "feiyu-rvc-release-v1"
        rvc_script = package / ("run_rvc.py" if public_rvc else "run_cover.py")
        rvc_prefix = []
        if cfg.get("rvc_runtime_profile"):
            if not public_rvc:
                raise ValueError("Managed runtime requires public FeiyuVoice-RVC v1")
            profile = load(resolve(cfg["rvc_runtime_profile"]))
            from feiyu_runtime.common import validate_profile
            validate_profile(profile)
            if profile["engine"] != "rvc":
                raise ValueError("RVC runtime profile engine mismatch")
            save(folder / "rvc_runtime_profile.json", profile)
            cfg["rvc_runtime"] = profile
            rvc_script = ROOT / "feiyu_runtime/worker.py"
            rvc_prefix = ["--kind", "rvc", "--package", linux_path(package),
                          "--profile", linux_path(folder / "rvc_runtime_profile.json")]
        command = runtime_command(cfg["rvc_runtime"], rvc_script, *rvc_prefix,
                                  "--input", linux_path(svs / "generated.wav"), "--output-dir", linux_path(rvc))
        if not public_rvc:
            command += ["--project-root", linux_path(ROOT)]
        # D76 cues are bound to the converted vocal, exactly as in the final package.
        if regions:
            command += ["--regions", linux_path(inputs / "regions.json")]
        run_logged(command, folder / "rvc_execution.log")
        cover = load(rvc / "cover_result.json")
        if cover["status"] != "success" or cover["package"] != manifest["id"]:
            raise ValueError("最终 RVC 包未成功完成转换。")
        result["rvc"] = cover
        native = rvc / "rvc_native.wav"
        if sha(native) != cover["native_sha256"]:
            raise ValueError("RVC 原生输出校验失败。")
        applied = cover.get("breath_attenuation_applied", cover.get("D76_applied", False))
        source = rvc / ("breath/final.wav" if public_rvc else "D76/final.wav") if applied else native
        result["audio"] = finalize_audio(source, folder / "singing.wav", compiled["duration_seconds"])
        result.update(status="success", output_file=str(folder / "singing.wav"), output_sha256=sha(folder / "singing.wav"),
                      D76_status="已应用本曲人工区间" if applied else "未应用；可对 rvc/rvc_native.wav 标记区间后另行应用")
        save(folder / "result.json", result)
        make_preview(folder, compiled, result)
        print(f"完成：{folder / 'singing.wav'}", flush=True)
        return result
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        save(folder / "result.json", result)
        make_preview(folder, compiled, result)
        raise
