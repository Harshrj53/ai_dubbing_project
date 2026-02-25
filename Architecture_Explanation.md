# Architecture Explanation: AI Hindi Video Dubbing Pipeline

This document explains the technical rationale behind the model selections and the system design for the AI Hindi Video Dubbing Pipeline.

## 1. Model Selections & Rationale

### Speech Recognition: `faster-whisper`
- **Choice**: `faster-whisper` (specifically the `large-v3` or `base` models depending on VRAM).
- **Reasoning**: Traditional Whisper is slow. `faster-whisper` uses CTranslate2, which is up to 4x faster and uses significantly less memory through quantization (INT8/FP16). 
- **Tradeoff**: Higher speed vs. minimal loss in transcription accuracy compared to the original OpenAI implementation.

### Machine Translation: `Helsinki-NLP/opus-mt-en-hi` (Fallback) & `IndicTrans2` (Preferred)
- **Choice**: `IndicTrans2` is the architectural recommendation, with `MarianMT` as the lightweight baseline.
- **Reasoning**: `IndicTrans2` is state-of-the-art for Indian languages, trained specifically on Indic languages to handle cultural nuances and complex grammar better than generic models like Google Translate or older MarianMT models.
- **Tradeoff**: `IndicTrans2` is larger and requires more complex environment setup; `MarianMT` is "plug-and-play" via Hugging Face.

### Voice Cloning: `Coqui XTTS v2`
- **Choice**: `Coqui XTTS v2`.
- **Reasoning**: It offers remarkable zero-shot voice cloning. It requires only a short (6+ second) audio sample to clone a voice across languages. It supports Hindi natively and maintains the original speaker's prosody and tone.
- **Tradeoff**: Computationally expensive during inference compared to standard TTS, but essential for the "Voice Cloning" requirement.

### Lip Sync: `VideoReTalking` (Preferred) & `Wav2Lip` (Robust Fallback)
- **Choice**: `VideoReTalking`.
- **Reasoning**: Standard Wav2Lip often results in blurry mouth areas. `VideoReTalking` integrates face restoration (GFPGAN/CodeFormer) directly into the pipeline, resulting in significantly higher fidelity and realistic facial movements.
- **Tradeoff**: `VideoReTalking` is much slower than `Wav2Lip` and has stricter dependency requirements (specific PyTorch versions).

---

## 2. Technical Design & Tradeoffs

### Quality vs. Compute
| Component | Low Compute (Fast) | High Quality (Slow/Heavy) |
| :--- | :--- | :--- |
| **Transcription** | Whisper `tiny`/`base` | Whisper `large-v3` |
| **Translation** | MarianMT | IndicTrans2 |
| **TTS** | FastSpeech2 | **Coqui XTTS v2** |
| **Lip Sync** | Wav2Lip | **VideoReTalking** |

The pipeline is designed to be **modular**, allowing us to swap the "Fast" version for the "High Quality" version simply by changing the model ID or the function call.

### Modular Pipeline Flow
The pipeline is decoupled to ensure that failure in one stage (e.g., Lip Sync) doesn't lose the progress of previous stages (Transcription/Translation). Each stage produces a cached intermediate artifact.

---

## 3. Scaling Strategy for 500 Hours
As detailed in the `README.md`, scaling requires moving from a sequential script to a **distributed worker-based architecture**.

1.  **VAD Splitter**: Using Voice Activity Detection to split long videos into 10-second segments.
2.  **Stateless Workers**: Each segment is processed independently on different GPU nodes.
3.  **Quantization**: Forcing INT8 quantization on all models to double the batch size per GPU.
4.  **Concatenation**: Re-stitching the 10-second segments into the final long-form video using `ffmpeg`'s concat demuxer.
