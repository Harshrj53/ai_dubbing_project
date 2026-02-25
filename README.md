# AI Hindi Video Dubbing Pipeline

This project provides a robust, fully open-source Python pipeline designed to translate an English training video and produce a high-quality Hindi-dubbed execution. It tightly integrates voice cloning, contextual text translation, and high-fidelity lip sync capabilities—requiring **₹0 budget** by leveraging state-of-the-art open models.

---

## 🏗 Modular Pipeline Architecture

The pipeline uses `dub_video.py` to enforce a sequential processing pattern:
1. **Video Extraction (`ffmpeg`)**: Identifies and separates a precise 15-second snippet without re-encoding to preserve immediate fidelity.
2. **Speech Recognition (`faster-whisper`)**: Extremely compute-efficient framework for pulling out time-stamped English speech mapping. Chosen over standard OpenAI whisper due to 4x inference speed optimizations leveraging CTranslate2 on VRAM.
3. **Translation (`MarianMT` / `IndicTrans2`)**: Context-aware transformer processing. `Helsinki-NLP/opus-mt-en-hi` operates as an open-source fallback robust enough to run rapidly off the shelf. (For ultimate Indian language nuances, `IndicTrans2` is fully advocated in broader production instances).
4. **Voice Cloning / TTS (`Coqui XTTS v2`)**: Leading zero-shot open-source vocal cloning. XTTS v2 creates an acoustic map of the presenter's original voice using only the 15-second sample, projecting emotional depth entirely in Hindi.
5. **Lip Sync (`VideoReTalking` / `Wav2Lip`)**: Combines generated audio arrays with the face bounding-box over the video. `VideoReTalking` is strongly encouraged for its implementation of GFPGAN/CodeFormer that removes standard visual mouth-blur routines resulting in production-tier facial preservation.

## 💻 Setup Instructions (Colab / Kaggle / Local Compatible)

**Hardware Requirements:** Run on environments equipped with an Nvidia GPU (T4 is sufficient for lightweight inference operations; A10G/A100 required for the intense rendering inside *VideoReTalking* pipelines).

```bash
# 1. Install OS-level dependencies
sudo apt-get update && sudo apt-get install -y ffmpeg

# 2. Clone the repository
git clone <your-organization-repo>
cd AI-Dubbing-Pipeline

# 3. Secure Python Pipeline Dependencies
pip install -r requirements.txt
# Alternatively, manually install the specific models:
# pip install faster-whisper transformers torch torchvision torchaudio TTS 

# 4. Clone Wav2Lip locally to handle the Lip Sync step
git clone https://github.com/Rudrabha/Wav2Lip.git
pip install -r Wav2Lip/requirements.txt
# Notice: You must download the Wav2Lip GAN weights manually and save to Wav2Lip/checkpoints/wav2lip_gan.pth

# 5. Run the Modular Processing File
python dub_video.py --input sample_video.mp4 --output final_15s_dub.mp4
```

## ⚠️ Known Limitations & Workarounds
- **Audio Expansion Drift:** English natively uses fewer syllables to express thoughts than Hindi. A translated Hindi output can stretch past original timestamps, misaligning video.
  - *Fix for more time resources:* Implement adaptive audio time-stretching (`pyrubberband` integrated with PyTorch) to match the XTTS output exactly to the faster-whisper segment timing maps.
- **Audio Pollution / Music:** Background noise ruins XTTS v2 vocal mapping.
  - *Fix:* Insert U-Net Audio Separators (like `Demucs`) pre-transcription to isolate only vocal tracks.

---

## 🔥 Scaling Feature: Orchestrating an Overnight 500-Hour Backlog 

*Scaling this codebase linearly is impossible due to VRAM OOM (Out-of-Memory) limits on long context tasks and extensive monolithic compute blocking.* To process **~500 hours** overnight smoothly, the script must be transformed into a **Distributed Microservices Pipeline**.

### 1. Batching Strategy & GPU Memory Logic
**Silence Trimming & Chunking**: You cannot load a 1-hour video into `faster-whisper` or `XTTS v2` directly.
- Deploy **VAD (Voice Activity Detection)** models like Silero VAD. 
- Splice long audio files completely at moments of silence; creating independent workloads of 5 to 15-second sentences.
- **Dynamic Batching:** Group these 10-second items into uniform lengths. Execute batch tensors holding ~16 to ~32 sentences synchronously into `XTTS v2` caching, drastically elevating GPU thread utilization.

### 2. Parallelization Approach (Queue Driven)
De-couple the python script (`dub_video.py`) into multiple individual API servers communicating over a message broker (e.g. RabbitMQ or Apache Kafka/AWS SQS). 
- **Service A (CPU Intense)**: ffmpeg audio extraction & VAD chunking.
- **Service B (GPU)**: faster-whisper processing and IndicTrans translation.
- **Service C (Heavy GPU)**: XTTS v2 Synthesis.
- **Service D (Heavy GPU)**: Video Lip Syncing (Wav2Lip/VideoReTalking).
*This pipeline allows translation services to scream through compute queues without getting bottlenecked waiting for Lip Sync to frame-by-frame format the video.*

### 3. Distributed Processing via Clouds
- Set up **Kubernetes (EKS/GKE)** clusters utilizing Node Autoscalers. 
- Bid for massive pools of **Spot Instances** (e.g., AWS EC2 `g5.xlarge` arrays using A10G chips or `g4dn` arrays). Spot instances drop compute costs up to 80%.

### 4. Model Quantization Layer
- Implement `Int8` quantization across all PyTorch Transformer deployments (leveraging the exact `bitsandbytes` library) — this cuts the VRAM requirements by half allowing you to stack two concurrent inference runs natively on a single GPU.
- Use `vLLM` or NVIDIA `TensorRT` to optimize TTS operations, eliminating massive latency blocks.

### 5. Scale Storage & Cost Approximation
- **Storage:** Leverage AWS S3 object arrays connected internally by Amazon EFS (Elastic File System) so every Kubernetes node can concurrently read/write intermediate artifacts without transferring data over HTTP blocks natively.
- **Estimated Cost:**
   - On heavily optimized GPU Spot Instances, total rendering mapping takes ~3 to ~4 minutes of compute time per 1 minute of video.
   - Using $0.35/hr spot pricing, processing achieves a rate equivalent to **$0.02 - $0.05 per processed video minute**. Costing ~$600 to ~$1500 to bulk render 500 hours overnight.
