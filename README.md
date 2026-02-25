Supernan AI Dubbing: Technical Pipeline Summary
"Supernan AI Dubbing ek end-to-end modular pipeline hai jo English videos ko natural Hindi mein dub karta hai. Humne isme zero-cost open-source tools use kiye hain aur focus 'Voice Accuracy' aur 'Lip-Sync' par rakha hai."

🚀 Pipeline Stages
Stage 1: Precision Clipping (FFmpeg)
Sabse pehle original video se exactly 20-second ka segment extract kiya gaya.
Isse processing fast hoti hai aur project ki requirements (15-30s) puri hoti hain.
Stage 2: Denoised Audio Extraction (FFmpeg)
Original speaker ki voice ko extract kiya gaya.
afftdn (Adaptive Noise Reduction) aur highpass filters ka use kiya gaya taaki background noise hat jaye aur voice cloning ke liye 'Clean Source' mile.
Stage 3: High-Accuracy Transcription (OpenAI Whisper)
Whisper-medium model ka use karke English speech ko text mein badla gaya. Ye open-source models mein sabsay behtreen accuracy deta hai.
Stage 4: Natural Hindi Translation (IndicTrans2)
Google Translate ki jagah IndicTrans2 (AI4Bharat) use kiya gaya. Ye model Indian languages ke liye context-aware translation karta hai, jo ki 'nanny training' jaisa sensitive content ke liye bahut zaruri hai.
Stage 5: Smart Voice Cloning (Coqui XTTS v2)
XTTS v2 model use kiya gaya jo sirf 5-10 second ki reference audio se original speaker ki voice clone kar leta hai.
Smart Splitting: Text ko logical points (conjunctions) par toda gaya taaki AI "fumble" na kare.
Clarity Booster: Audio generate hone ke baad professional studio filters (Equalizer, Compressor, Loudnorm) lagaye gaye.
Stage 6: Natural Sync & Speed Locking (FFmpeg)
Audio ki speed ko video se match kiya gaya, lekin speed ko 1.15x tak limit rakha gaya taaki voice unnatural na lage.
Agar audio lambi ho jaye, toh humne 'Smart Padding' (last frame freeze) use kiya hai.
Stage 7: Robust Lip-Sync (VideoReTalking)
VideoReTalking model ka use karke video ke lip movements ko Hindi audio ke according update kiya gaya. Ye Wav2Lip se zyada sharp result deta hai.
🛠️ Tools & Models Used
FFmpeg: Audio extraction, denoising, speed adjustment, aur final muxing ke liye.
OpenAI Whisper (Medium): Transcription ke liye.
IndicTrans2 (AI4Bharat): Professional Hindi translation ke liye.
Coqui XTTS v2: Zero-shot voice cloning ke liye.
VideoReTalking: High-quality lip-syncing ke liye.
GFPGAN: Face restoration aur sharpening ke liye.
📓 Usage
For high-fidelity results with GPU acceleration (Face Restoration & Lip-Sync), use the provided Colab Notebook: Supernan_AI_Dubbing.ipynb
