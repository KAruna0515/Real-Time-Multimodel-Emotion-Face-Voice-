import cv2, sounddevice as sd, soundfile as sf, io, numpy as np, time
from app.inference import engine

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0

def record_chunk():
    rec = sd.rec(int(CHUNK_SECONDS*SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait(); data = rec.flatten()
    bio = io.BytesIO(); sf.write(bio, data, SAMPLE_RATE, format='WAV')
    return bio.getvalue()

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): print("Cannot open webcam"); return
    print("Local-mode running (press q to quit).")
    while True:
        ret, frame = cap.read()
        if not ret: break
        wav = record_chunk()
        _, buf = cv2.imencode('.jpg', frame)
        res = engine.analyze(wav_bytes=wav, frame_bytes=buf.tobytes())
        fused = res.get("fusion", {}).get("fused_probs")
        label = "-"
        if fused:
            arr = np.array(fused); idx = arr.argmax(); label = res.get("labels", [])[idx]
        cv2.putText(frame, f"Fused: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("Local Mode", frame)
        if cv2.waitKey(1)&0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
