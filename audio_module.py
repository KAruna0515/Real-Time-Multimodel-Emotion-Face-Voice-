import io
import logging
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from .config import get_device, EMOTION_LABELS, USE_WHISPER_ASR

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MFCC = 40
HOP_LENGTH = 160
WIN_LENGTH = 400

def load_wav_bytes(wav_bytes: bytes, sr=SAMPLE_RATE) -> np.ndarray:
    bio = io.BytesIO(wav_bytes)
    data, fs = sf.read(bio, dtype="float32")
    if fs != sr:
        data = librosa.resample(data, orig_sr=fs, target_sr=sr)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data

def extract_mfcc(y: np.ndarray, sr=SAMPLE_RATE, n_mfcc=N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
    return mfcc.astype("float32")

def pad_or_trim(feature: np.ndarray, max_frames=200) -> np.ndarray:
    n_mfcc, T = feature.shape
    if T >= max_frames:
        return feature[:, :max_frames]
    out = np.zeros((n_mfcc, max_frames), dtype=feature.dtype)
    out[:, :T] = feature
    return out

class AudioEmotionModel(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, hidden=64):
        super().__init__()
        n_classes = len(EMOTION_LABELS)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.rnn = nn.LSTM(input_size=(n_mfcc//4)*32, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self,x):
        b = x.size(0)
        x = self.cnn(x)
        _, c, nf, t = x.size()
        x = x.permute(0,3,1,2).contiguous()
        x = x.view(b, t, -1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class AudioPipeline:
    def __init__(self, n_classes: Optional[int] = None):
        self.device = get_device()
        # n_classes is now determined by the model itself from config
        self.model = AudioEmotionModel().to(self.device)
        self.model.eval()
        # ASR pipeline optional
        self.asr = None
        if USE_WHISPER_ASR:
            try:
                from transformers import pipeline
                device_id = self.device.index if self.device.type == "cuda" else -1
                self.asr = pipeline("automatic-speech-recognition",
                                    model="openai/whisper-tiny",
                                    device=device_id)
            except Exception as e:
                logger.warning(f"ASR pipeline failed to load: {e}")
                self.asr = None

    def predict(self, wav_bytes: bytes) -> Dict[str, Any]:
        y = load_wav_bytes(wav_bytes)
        mfcc = extract_mfcc(y)
        mfcc = pad_or_trim(mfcc, max_frames=200)
        arr = mfcc[np.newaxis, np.newaxis, ...]
        tensor = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()

        transcript = None
        if self.asr:
            try:
                out = self.asr(y, sampling_rate=SAMPLE_RATE)
                transcript = out.get("text")
            except Exception:
                transcript = None
        return {"probs": probs, "transcript": transcript}
