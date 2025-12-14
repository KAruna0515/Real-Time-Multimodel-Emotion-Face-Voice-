import threading, time, io, os
import cv2, requests, sounddevice as sd, soundfile as sf, numpy as np
from collections import deque
from .live_camera_client import __name__  # silence unused in packaging

API_URL = os.getenv("MM_API_URL", "http://127.0.0.1:8000/analyze")
SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 2.0
SEND_INTERVAL = 2.0
FRAME_W = 640; FRAME_H = 480

class AudioRecorder:
    def __init__(self, sr=SAMPLE_RATE, secs=AUDIO_CHUNK_SECONDS):
        self.sr = sr
        self.chunk = int(sr*secs)
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=self.chunk)
        self.stream = None
    def _cb(self, indata, frames, timeinfo, status):
        if status: print("Audio status:", status)
        data = indata.copy()
        if data.ndim > 1: data = data.mean(axis=1)
        with self.lock:
            for v in data: self.buffer.append(float(v))
    def start(self):
        self.buffer.clear()
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', callback=self._cb, blocksize=0)
        self.stream.start()
    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close()
    def get_wav(self):
        with self.lock:
            samples = list(self.buffer)
        if len(samples) < self.chunk:
            pad = [0.0]*(self.chunk - len(samples)); samples = pad + samples
        else:
            samples = samples[-self.chunk:]
        arr = np.array(samples, dtype='float32')
        bio = io.BytesIO(); sf.write(bio, arr, self.sr, format='WAV'); return bio.getvalue()

def send_to_api(frame_bgr, wav_bytes, api_url=API_URL, timeout=20):
    try:
        _, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY),80])
        files = {"audio":("chunk.wav", wav_bytes, "audio/wav"), "frame":("frame.jpg", buf.tobytes(), "image/jpeg")}
        r = requests.post(api_url, files=files, timeout=timeout); r.raise_for_status(); return r.json()
    except Exception as e:
        print("API error:", e); return None

def overlay(frame, res):
    overlay = frame.copy(); h,w=frame.shape[:2]
    cv2.rectangle(overlay,(0,0),(w,90),(0,0,0),-1); alpha=0.45; cv2.addWeighted(overlay, alpha, frame, 1-alpha,0,frame)
    if not res:
        cv2.putText(frame,"No result",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2); return frame
    fused_top = res.get("fusion",{}).get("fused_probs")
    transcript = res.get("audio",{}).get("transcript")
    label_text = "Fused: -"
    if fused_top:
        arr = np.array(fused_top); idx = arr.argsort()[::-1][0]; lab = res.get("labels",[])[idx]; score = arr[idx]; label_text = f"Fused: {lab} ({score:.2f})"
    cv2.putText(frame,label_text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    if transcript:
        txt = transcript[:60]
        cv2.putText(frame,f"ASR: {txt}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
    return frame

def live_loop():
    recorder = AudioRecorder(); recorder.start()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened(): print("Cannot open webcam"); recorder.stop(); return
    last_send=0; last_res=None
    print("Live client started. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            now=time.time()
            if now - last_send >= SEND_INTERVAL:
                wav = recorder.get_wav(); frame_copy = frame.copy()
                def worker(f,a):
                    nonlocal last_res
                    r = send_to_api(f,a)
                    if r is not None: last_res = r
                threading.Thread(target=worker, args=(frame_copy,wav), daemon=True).start()
                last_send = now
            display = overlay(frame, last_res)
            cv2.imshow("Live Multimodal Emotion", display)
            if cv2.waitKey(1)&0xFF == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows(); recorder.stop()

if __name__ == "__main__":
    live_loop()
