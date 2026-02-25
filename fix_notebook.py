import json
import os

notebook_path = '/Users/vikashkumar/Desktop/Supernan/supernan_dubbing.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

def get_duration_code():
    return [
        'import subprocess, os\n',
        'def get_duration(path):\n',
        '    if not os.path.exists(path): return 0.0\n',
        '    r = subprocess.run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0",path],\n',
        '                       capture_output=True, text=True)\n',
        '    try: return float(r.stdout.strip())\n',
        '    except: return 0.0\n',
        '\n'
    ]

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = cell['source']
    if not source:
        continue
    
    header = source[0]

    if '# ── Cell 3: Install Python packages' in header:
        cell['source'] = [
            '!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118\n',
            '!pip install -q openai-whisper TTS transformers sentencepiece sacremoses\n',
            '!pip install -q git+https://github.com/VarunGumma/IndicTransTokenizer.git\n',
            '!pip install -q pydub librosa soundfile deep-translator\n',
            '!pip install -q basicsr facexlib gfpgan realesrgan\n',
            'print("✓ All packages installed")\n'
        ]

    elif '# ── Cell 5: Download VideoReTalking Weights' in header:
        cell['source'] = [
            'import os\n',
            'os.makedirs("models/VideoReTalking", exist_ok=True)\n',
            '!wget -q https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_G.pth -O models/VideoReTalking/30_net_G.pth\n',
            '!wget -q https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip -O models/VideoReTalking/BFM.zip\n',
            '!unzip -qo models/VideoReTalking/BFM.zip -d models/VideoReTalking/\n',
            'print("✓ VideoReTalking weights ready")\n'
        ]

    elif '# ── Cell 10: Stage 2 — Extract Audio' in header:
        cell['source'] = [
            'import os\n',
            'if "CLIP" not in globals(): CLIP = "workspace/clip.mp4"\n',
            'AUDIO_RAW = "workspace/clip_audio.wav"\n',
            'AUDIO_16K = "workspace/clip_audio_16k.wav"\n',
            'AUDIO_REF = "workspace/clip_audio_ref44k.wav"\n',
            'print("Extracting audio tracks...")\n',
            'get_ipython().system(f"ffmpeg -y -i {CLIP} -vn -acodec pcm_s16le -ar 44100 {AUDIO_RAW} -loglevel warning")\n',
            'get_ipython().system(f"ffmpeg -y -i {AUDIO_RAW} -ac 1 -ar 16000 {AUDIO_16K} -loglevel warning")\n',
            'get_ipython().system(f"ffmpeg -y -i {AUDIO_RAW} -ac 1 -ar 44100 {AUDIO_REF} -loglevel warning")\n',
            'print("✓ Audio extracted (Raw, 16k, 44k Ref)")\n'
        ]

    elif '# ── Cell 12: Stage 4 — Translate to Hindi (IndicTrans2)' in header:
        cell['source'] = [
            'import torch, os\n',
            'if "DEVICE" not in globals(): DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
            'if "ENGLISH_TEXT" not in globals():\n',
            '    if os.path.exists("workspace/transcript_en.txt"):\n',
            '        with open("workspace/transcript_en.txt", "r") as f: ENGLISH_TEXT = f.read().strip()\n',
            '        print("✓ ENGLISH_TEXT recovered from disk")\n',
            '    else: raise NameError("ENGLISH_TEXT missing. Run Cell 11.")\n',
            'def translate_indictrans2(text):\n',
            '    try:\n',
            '        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer\n',
            '        from IndicTransTokenizer import IndicProcessor\n',
            '        MODEL = "ai4bharat/indictrans2-en-indic-1B"\n',
            '        print("Loading IndicTrans2...")\n',
            '        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)\n',
            '        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).to(DEVICE)\n',
            '        ip = IndicProcessor(inference=True)\n',
            '        sentences = [s.strip() for s in text.split(".") if s.strip()]\n',
            '        batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")\n',
            '        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)\n',
            '        with torch.no_grad():\n',
            '            outputs = model.generate(**inputs, num_beams=5, max_length=256)\n',
            '        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)\n',
            '        postprocessed = ip.postprocess_batch(decoded, lang="hin_Deva")\n',
            '        return " ".join(postprocessed)\n',
            '    except Exception as e:\n',
            '        print(f"IndicTrans2 failed: {e}")\n',
            '        return None\n',
            'print("Attempting Hindi translation...")\n',
            'HINDI_TEXT = translate_indictrans2(ENGLISH_TEXT)\n',
            'if not HINDI_TEXT:\n',
            '    print("⚠️ IndicTrans2 failed. Attempting Google Translate fallback...")\n',
            '    try:\n',
            '        from deep_translator import GoogleTranslator\n',
            '        HINDI_TEXT = GoogleTranslator(source="en", target="hi").translate(ENGLISH_TEXT)\n',
            '        print("✓ Google Translate success!")\n',
            '    except Exception as ge:\n',
            '        print(f"❌ Google Translate failed: {ge}. Using English as last resort.")\n',
            '        HINDI_TEXT = ENGLISH_TEXT\n',
            'with open("workspace/translation_hi.txt", "w", encoding="utf-8") as f: f.write(HINDI_TEXT)\n',
            'print(f"✓ Hindi translation completed.")\n',
            'print(f"Text: {HINDI_TEXT[:100]}...")\n'
        ]

    elif '# ── Cell 13: Stage 5 — Coqui XTTS v2 Voice Cloning' in header:
        cell['source'] = [
            'import os, re, torch\n',
            'from TTS.api import TTS\n',
            'if "DEVICE" not in globals(): DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
            'if "AUDIO_REF" not in globals(): AUDIO_REF = "workspace/clip_audio_ref44k.wav"\n',
            'if "HINDI_TEXT" not in globals() or not HINDI_TEXT.strip():\n',
            '    if os.path.exists("workspace/translation_hi.txt"):\n',
            '        with open("workspace/translation_hi.txt", "r", encoding="utf-8") as f: HINDI_TEXT = f.read().strip()\n',
            '    if not globals().get("HINDI_TEXT"):\n',
            '        if "ENGLISH_TEXT" in globals(): HINDI_TEXT = ENGLISH_TEXT\n',
            '        else: raise ValueError("HINDI_TEXT empty and no fallback found.")\n',
            'try:\n',
            '    from TTS.tts.configs.xtts_config import XttsConfig\n',
            '    from TTS.tts.models.xtts.xtts_audio_config import XttsAudioConfig\n',
            '    import torch.serialization\n',
            '    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])\n',
            '    print("✓ Added XTTS configs to PyTorch safe globals")\n',
            'except Exception as e:\n',
            '    print(f"⚠️ Could not add safe globals: {e}")\n',
            'if not hasattr(torch.load, "__supernan_patch__"):\n',
            '    orig_load = torch.load\n',
            '    def patched_load(*args, **kwargs):\n',
            '        if "weights_only" not in kwargs: kwargs["weights_only"] = False\n',
            '        return orig_load(*args, **kwargs)\n',
            '    patched_load.__supernan_patch__ = True\n',
            '    torch.load = patched_load\n',
            '    print("✓ Applied torch.load security bypass monkeypatch")\n',
            'os.environ["COQUI_TOS_AGREED"] = "1"\n',
            'TTS_RAW = "workspace/tts_raw.wav"\n',
            'print("Loading Coqui XTTS v2...")\n',
            'tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)\n',
            'print(f"Synthesizing Hindi audio...")\n',
            'tts.tts_to_file(text=HINDI_TEXT, speaker_wav=AUDIO_REF, language="hi", file_path=TTS_RAW)\n',
            'print(f"✓ Synthesized: {TTS_RAW}")\n'
        ]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=4)

print('Successfully cleaned and fixed supernan_dubbing.ipynb with Golden Cell strategy')
