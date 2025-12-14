# app/config.py
import os
from dotenv import load_dotenv
import torch

load_dotenv()

DEVICE = os.getenv("DEVICE", "cpu").lower()
USE_WHISPER_ASR = os.getenv("USE_WHISPER_ASR", "False").lower() == "true"
EMOTION_LABELS = [s.strip() for s in os.getenv("EMOTION_LABELS",
    "neutral,happy,sad,angry,fear,disgust,surprise").split(",")]

def get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
