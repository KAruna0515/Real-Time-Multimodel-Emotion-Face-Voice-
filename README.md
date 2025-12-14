# Real-Time Multimodal Emotion AI (Voice + Face)

Live camera + microphone multimodal emotion detection (audio + face + optional text ASR).
Project includes:
- FastAPI server (app/server.py) exposing `/analyze`
- Singleton engine (app/inference.py) combining audio + vision + text pipelines
- Demo clients:
  - demo/live_camera_client.py (network client; sends webcam frame + audio chunk to server)
  - demo/localmode_client.py (no-network; calls engine directly for ultra-low latency)
  - demo/streamlit_ui.py (optional web UI using Streamlit)
- Windows-friendly scripts in `scripts/`

## Quickstart (Windows)
1. Open PowerShell and run (one line at a time):
```powershell
Set-Location "path\to\realtime-multimodal-emotion-ai"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
# edit .env if needed (DEVICE=cpu recommended)
