import os
import torch

START_SEC = 15
END_SEC = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

VIDEORETALKING_DIR = os.path.join(BASE_DIR, "VideoReTalking")
GFPGAN_DIR = os.path.join(BASE_DIR, "GFPGAN")

WHISPER_MODEL_SIZE = "medium"

INDICTRANS_MODEL = "ai4bharat/indictrans2-en-indic-1B"
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "hin_Deva"

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE = "hi"
TTS_SAMPLE_RATE = 24000

SILENCE_THRESH_DB = -40
MIN_SILENCE_MS = 500
KEEP_SILENCE_MS = 250

ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

FACE_DET_THRESH = 0.9
VRT_USE_ENHANCER = False

GFPGAN_VERSION = "1.4"
GFPGAN_UPSCALE = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_LEVEL = "INFO"

for _d in [WORKSPACE_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)
