import os
import subprocess
import logging
from typing import List, Tuple, Dict

# Setting up basic logging for clear console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command: str) -> None:
    """Utility to run shell commands (like ffmpeg or external repos)."""
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {command}")
        if e.stderr:
            logging.error(e.stderr.decode('utf-8'))
        raise

def extract_clip(input_video: str, output_clip: str, start_time: str = "00:00:15", duration: str = "00:00:15") -> str:
    """
    Extracts a precise 15-second clip from the input video using ffmpeg.
    """
    logging.info(f"Extracting a {duration} clip starting from {start_time}")
    cmd = f"ffmpeg -y -i {input_video} -ss {start_time} -t {duration} -c:v copy -c:a copy {output_clip}"
    run_command(cmd)
    return output_clip

def _extract_audio(video_file: str, output_audio: str) -> str:
    """Helper method to isolate the audio stream from the video."""
    logging.info("Extracting audio stream from the video...")
    cmd = f"ffmpeg -y -i {video_file} -q:a 0 -map a {output_audio}"
    run_command(cmd)
    return output_audio

def transcribe_audio(audio_file: str) -> Tuple[List[Dict], str]:
    """
    Extracts audio and transcribes English speech using `faster-whisper`.
    Faster-whisper prioritizes inference speed and lower VRAM usage.
    Maintains timestamps.
    """
    logging.info("Transcribing audio using faster-whisper...")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logging.error("faster-whisper is not installed. Please install it using `pip install faster-whisper`")
        raise
        
    model_size = "base" # Can be swapped to "large-v3" for high fidelity logic
    # Running on GPU, heavily utilizing FP16. Change device to "cpu" if no GPU available.
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    segments, info = model.transcribe(audio_file, beam_size=5)
    
    transcript = []
    full_text = ""
    for segment in segments:
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        full_text += segment.text + " "
        
    logging.info(f"Transcription completed. Detected language probability: {info.language_probability}")
    return transcript, full_text.strip()

def translate_text(text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
    """
    Translates the English transcript contextually into Hindi.
    Utilizes Hugging Face transformers. Note: IndicTrans2 is preferred conceptually for Indic languages,
    but here we wrap MarianMT as the robust, instantly-deployable open-source fallback.
    """
    logging.info("Translating text contextually from English to Hindi...")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        logging.error("transformers is not installed. Please install it using `pip install transformers torch`")
        raise

    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
    
    # Process inputs (can be expanded for chunked batching in scaling setups)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    logging.info(f"Final Translation: {translated_text}")
    return translated_text

def generate_voice(target_text: str, reference_audio: str, output_audio: str, language: str = "hi") -> str:
    """
    Generates natural Hindi speech that matches the original speaker's tone.
    Uses Coqui XTTS v2 for zero-shot voice cloning.
    """
    logging.info("Generating localized voice clone using Coqui XTTS v2...")
    try:
        from TTS.api import TTS
    except ImportError:
        logging.error("TTS is not installed. Please install it using `pip install TTS`")
        raise

    # Model is quite heavy, typically run on GPU
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    
    # TTS generation using original audio as voice conditioning profile
    tts.tts_to_file(
        text=target_text,
        speaker_wav=reference_audio,
        language=language,
        file_path=output_audio
    )
    return output_audio

def silence_trim_audio(input_audio: str, output_audio: str) -> str:
    """
    Removes long silences from the audio to ensure the TTS doesn't waste compute 
    or result in awkward pauses in the dubbed version.
    """
    logging.info(f"Trimming silence from {input_audio}")
    # ffmpeg silence removal filter
    cmd = f"ffmpeg -y -i {input_audio} -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-30dB {output_audio}"
    run_command(cmd)
    return output_audio

def lipsync_video(video_file: str, audio_file: str, output_video: str) -> str:
    """
    Syncs the generated Hindi audio with the underlying video. 
    VideoReTalking is strongly preferred for high fidelity and avoiding blurry faces,
    but this logic wraps Wav2Lip as an easier-to-execute fallback locally.
    """
    logging.info("Starting Video Lip Sync Process...")
    
    # Check for Wav2Lip fallback repository
    wav2lip_dir = "./Wav2Lip"
    checkpoint = f"{wav2lip_dir}/checkpoints/wav2lip_gan.pth"
    
    if os.path.exists(wav2lip_dir):
        logging.info("Wav2Lip repository found, executing lip-sync inference script.")
        cmd = f"python {wav2lip_dir}/inference.py --checkpoint_path {checkpoint} --face {video_file} --audio {audio_file} --outfile {output_video}"
        run_command(cmd)
    else:
        logging.warning("Wav2Lip or VideoReTalking directory not found locally!")
        logging.warning("Skipping intensive lip-sync generation. Yielding audio-merged video fallback...")
        merge_output(video_file, audio_file, output_video)
        
    return output_video

def merge_output(video_file: str, audio_file: str, final_output: str) -> str:
    """
    Combines the source video layer with the generated Hindi audio layer.
    """
    logging.info("Merging final audio with video...")
    cmd = f"ffmpeg -y -i {video_file} -i {audio_file} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {final_output}"
    run_command(cmd)
    return final_output

def _chunk_segments_for_scaling(transcript: List[Dict], max_chunk_duration: int = 10) -> List[List[Dict]]:
    """
    Splits the transcription into small, manageable chunks for parallel processing or 
    to prevent GPU VRAM exhaustion on long videos.
    """
    logging.info(f"Chunking transcript for scalability (Max chunk: {max_chunk_duration}s)")
    chunks = []
    current_chunk = []
    chunk_start = transcript[0]['start'] if transcript else 0
    
    for segment in transcript:
        if segment['end'] - chunk_start > max_chunk_duration:
            chunks.append(current_chunk)
            current_chunk = [segment]
            chunk_start = segment['start']
        else:
            current_chunk.append(segment)
            
    if current_chunk:
        chunks.append(current_chunk)
        
    logging.info(f"Split video into {len(chunks)} processing chunks.")
    return chunks

def process_video(input_video: str, final_output: str = "output_dubbed.mp4") -> None:
    """
    Primary orchestrator executing the end-to-end AI Hindi Dubbing workflow.
    """
    # Define intermediary tracking files
    clip_vid = "temp_15s_clip.mp4"
    orig_aud = "temp_orig_audio.wav"
    hindi_aud = "temp_hindi_audio.wav"

    try:
        logging.info("Starting Modular AI Video Dubbing Pipeline")
        
        # 1. 15-second extraction
        extract_clip(input_video, clip_vid)
        
        # 2. Extract embedded audio
        _extract_audio(clip_vid, orig_aud)
        
        # 3. Transcribe base speech
        transcript, text_en = transcribe_audio(orig_aud)
        _chunk_segments_for_scaling(transcript)  # Structural placeholder for handling >15s videos
        
        # 4. Neural Translation
        text_hi = translate_text(text_en, source_lang="en", target_lang="hi")
        
        # 5. Native GenAI TTS
        generate_voice(text_hi, orig_aud, hindi_aud, language="hi")
        
        # 6. Model Output Syncing
        lipsync_video(clip_vid, hindi_aud, final_output)
        
        logging.info(f"Pipeline finished! Final output is successfully located at {final_output}")
        
    except Exception as e:
        logging.error(f"Pipeline crashed during execution: {str(e)}")
    finally:
        # Prevent disk bloating; strict artifact garbage collection
        for temp_file in [clip_vid, orig_aud, hindi_aud]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logging.info(f"Cleaned up intermediate file: {temp_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Hindi Video Dubbing Pipeline")
    parser.add_argument("--input", required=True, help="Path to the input MP4 video file")
    parser.add_argument("--output", default="output_dubbed.mp4", help="Path to the output dubbed video file")
    args = parser.parse_args()
    
    process_video(args.input, args.output)
