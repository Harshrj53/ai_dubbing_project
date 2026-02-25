#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import shutil
import subprocess
from pathlib import Path

from config import (
    START_SEC, END_SEC, WORKSPACE_DIR, OUTPUT_DIR, MODELS_DIR,
    VIDEORETALKING_DIR, GFPGAN_DIR,
    WHISPER_MODEL_SIZE,
    INDICTRANS_MODEL, SOURCE_LANG, TARGET_LANG,
    XTTS_MODEL_NAME, TTS_LANGUAGE, TTS_SAMPLE_RATE,
    SILENCE_THRESH_DB, MIN_SILENCE_MS, KEEP_SILENCE_MS,
    FACE_DET_THRESH, VRT_USE_ENHANCER,
    GFPGAN_VERSION, GFPGAN_UPSCALE,
    DEVICE,
)
from utils import (
    run_cmd, get_duration,
    stretch_audio,
    split_on_silence, concatenate_wavs,
    check_sync_offset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(WORKSPACE_DIR, "pipeline.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

def extract_clip(input_path: str, output_path: str, start: float, end: float) -> str:
    duration = end - start
    logger.info(f"[Stage 1] Extracting clip: {start}s – {end}s ({duration}s)")
    run_cmd([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "18",
        output_path
    ], desc="Extracting clip segment")
    logger.info(f"  ✓ Clip saved: {output_path}")
    return output_path

def extract_audio(video_path: str, output_wav: str) -> str:
    logger.info("[Stage 2] Extracting audio")
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav
    ], desc="Extracting audio as 16kHz mono WAV")

    ref_wav = output_wav.replace(".wav", "_ref44k.wav")
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        ref_wav
    ], desc="Extracting 44kHz reference audio for voice cloning")

    logger.info(f"  ✓ Audio (16kHz): {output_wav}")
    logger.info(f"  ✓ Reference (44kHz): {ref_wav}")
    return output_wav

def transcribe(audio_path: str, transcript_path: str) -> str:
    import whisper
    logger.info(f"[Stage 3] Transcribing with Whisper ({WHISPER_MODEL_SIZE})")
    model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)

    result = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        verbose=False,
    )
    text = result["text"].strip()

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    logger.info(f"  ✓ Transcript: {text[:100]}{'...' if len(text) > 100 else ''}")
    logger.info(f"  ✓ Saved to: {transcript_path}")
    return text

def translate(english_text: str, translation_path: str) -> str:
    hindi_text = _translate_indictrans2(english_text)
    if hindi_text is None:
        logger.warning("  IndicTrans2 unavailable. Falling back to deep-translator...")
        hindi_text = _translate_fallback(english_text)

    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(hindi_text + "\n")

    logger.info(f"  ✓ Hindi: {hindi_text[:100]}{'...' if len(hindi_text) > 100 else ''}")
    logger.info(f"  ✓ Saved to: {translation_path}")
    return hindi_text

def _translate_indictrans2(text: str) -> str | None:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransTokenizer import IndicProcessor

        logger.info(f"[Stage 4] Loading IndicTrans2 ({INDICTRANS_MODEL})")
        tokenizer = AutoTokenizer.from_pretrained(
            INDICTRANS_MODEL, trust_remote_code=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICTRANS_MODEL, trust_remote_code=True
        ).to(DEVICE)
        ip = IndicProcessor(inference=True)

        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        BATCH_SIZE = 10
        translations = []

        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]
            batch_input = ip.preprocess_batch(
                batch, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG
            )
            inputs = tokenizer(
                batch_input,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)
            with __import__("torch").no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=5,
                    num_return_sequences=1,
                    max_length=256,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            postprocessed = ip.postprocess_batch(decoded, lang=TARGET_LANG)
            translations.extend(postprocessed)

        return " ".join(translations)

    except (ImportError, Exception):
        return None

def _translate_fallback(text: str) -> str:
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="en", target="hi").translate(text)

STAGE_ORDER = [
    "extract_clip",
    "extract_audio",
    "transcribe",
    "translate",
    "synthesize_voice",
    "adjust_audio_speed",
    "lipsync",
    "enhance_face",
]

def main():
    parser = argparse.ArgumentParser(
        description="Supernan Hindi Dubbing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--start", "-s", type=float, default=START_SEC)
    parser.add_argument("--end", "-e", type=float, default=END_SEC)
    parser.add_argument("--skip-lipsync", action="store_true")
    parser.add_argument("--skip-enhance", action="store_true")
    parser.add_argument("--from-stage", metavar="STAGE", choices=STAGE_ORDER)
    parser.add_argument("--check-sync-only", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    if args.end <= args.start:
        parser.error(f"--end ({args.end}) must be greater than --start ({args.start})")

if __name__ == "__main__":
    main()
