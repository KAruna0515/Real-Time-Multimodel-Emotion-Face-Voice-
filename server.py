from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .inference import engine

app = FastAPI(title="Real-Time Multimodal Emotion AI", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...), frame: UploadFile = File(...)):
    wav_bytes = await audio.read()
    frame_bytes = await frame.read()
    try:
        result = await engine.analyze(wav_bytes=wav_bytes, frame_bytes=frame_bytes)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
