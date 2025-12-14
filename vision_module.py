# app/vision_module.py
import cv2, numpy as np, io
import torch, torch.nn as nn
from .config import get_device

class FaceEmotionModel(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*56*56,128), nn.ReLU(), nn.Linear(128,n_classes)
        )
    def forward(self,x): return self.net(x)

class VisionPipeline:
    def __init__(self,n_classes=7):
        self.device = get_device()
        self.model = FaceEmotionModel(n_classes=n_classes).to(self.device)
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    def detect_face(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64,64))
        if len(faces)==0: return None
        x,y,w,h = faces[0]; return gray[y:y+h, x:x+w]
    def predict(self, frame_bgr):
        face = self.detect_face(frame_bgr)
        if face is None: return {"probs": None, "has_face": False}
        face_resized = cv2.resize(face, (224,224)).astype("float32")/255.0
        arr = face_resized[np.newaxis, np.newaxis, ...]
        t = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            logits = self.model(t); probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return {"probs": probs.tolist(), "has_face": True}
    def predict_from_bytes(self, frame_bytes):
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.predict(frame_bgr)
