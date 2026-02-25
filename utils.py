import os
import logging
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def run_cmd(cmd: List[str], desc: str = "", check: bool = True) -> subprocess.CompletedProcess:
    if desc:
        logger.info(f"[CMD] {desc}")
    logger.debug(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stdout:
        logger.debug(f"  stdout: {result.stdout.strip()}")
    if result.stderr:
        logger.debug(f"  stderr: {result.stderr.strip()}")
    if check and result.returncode != 0:
        logger.error(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
        logger.error(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    return result


def get_duration(filepath: str) -> float:
    result = run_cmd([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        filepath
    ], desc=f"Probing duration of {os.path.basename(filepath)}")
    return float(result.stdout.strip())


def _build_atempo_chain(ratio: float) -> str:
    filters = []
    remaining = ratio
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def stretch_audio(
    input_path: str,
    output_path: str,
    target_duration: float,
) -> str:
    src_duration = get_duration(input_path)
    ratio = src_duration / target_duration
    atempo_str = _build_atempo_chain(ratio)

    logger.info(
        f"Stretching audio: {src_duration:.2f}s → {target_duration:.2f}s "
        f"(ratio={ratio:.3f}, filter={atempo_str})"
    )

    run_cmd([
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", atempo_str,
        "-ar", "44100",
        output_path
    ], desc="Stretching audio to match video duration")

    return output_path


def split_on_silence(
    input_wav: str,
    output_dir: str,
    silence_thresh_db: float = -40,
    min_silence_ms: int = 500,
    keep_silence_ms: int = 250,
) -> List[str]:

    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence as _split
    except ImportError:
        raise ImportError("pydub is required for silence splitting. Install it: pip install pydub")

    logger.info(f"Splitting audio on silence: {os.path.basename(input_wav)}")
    audio = AudioSegment.from_wav(input_wav)

    chunks = _split(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms,
    )

    logger.info(f"  → {len(chunks)} chunk(s) detected")

    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []

    for i, chunk in enumerate(chunks):
        path = os.path.join(output_dir, f"chunk_{i:04d}.wav")
        chunk.export(path, format="wav")
        chunk_paths.append(path)
        logger.debug(f"  Wrote chunk {i}: {len(chunk)/1000:.2f}s → {path}")

    return sorted(chunk_paths)


def concatenate_wavs(wav_paths: List[str], output_path: str) -> str:
    if not wav_paths:
        raise ValueError("No WAV files provided to concatenate")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in wav_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        concat_file = f.name

    try:
        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ], desc="Concatenating audio chunks")
    finally:
        os.unlink(concat_file)

    return output_path


def check_sync_offset(
    reference_wav: str,
    dubbed_wav: str,
    sample_rate: int = 16000,
) -> Tuple[float, float]:

    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for sync check. Install: pip install librosa")

    ref, _ = librosa.load(reference_wav, sr=sample_rate, mono=True)
    dub, _ = librosa.load(dubbed_wav, sr=sample_rate, mono=True)

    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    dub = dub / (np.max(np.abs(dub)) + 1e-8)

    corr = np.correlate(ref, dub, mode="full")
    lag = np.argmax(np.abs(corr)) - (len(dub) - 1)
    offset_sec = lag / sample_rate
    confidence = float(np.max(np.abs(corr)) / len(ref))

    logger.info(f"Sync check → offset={offset_sec*1000:.1f}ms, confidence={confidence:.3f}")
    return offset_sec, confidence


def download_file(url: str, dest_path: str, desc: str = "") -> str:
    if os.path.exists(dest_path):
        logger.info(f"Already exists, skipping download: {os.path.basename(dest_path)}")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    logger.info(f"Downloading {desc or url} → {dest_path}")

    run_cmd(
        ["wget", "-q", "--show-progress", "-O", dest_path, url],
        desc=f"Downloading {desc}"
    )

    return dest_path
