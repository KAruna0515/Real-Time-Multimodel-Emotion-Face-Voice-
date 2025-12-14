# app/inference.py
import asyncio
from .audio_module import AudioPipeline
from .vision_module import VisionPipeline
from .text_module import TextEmotionPipeline
from .fusion import fuse_probabilities
from .config import EMOTION_LABELS

class MultimodalEmotionEngine:
    def __init__(self):
        n = len(EMOTION_LABELS)
        self.audio = AudioPipeline(n_classes=n)
        self.vision = VisionPipeline(n_classes=n)
        try:
            self.text = TextEmotionPipeline()
        except Exception:
            self.text = None

    async def analyze(self, wav_bytes=None, frame_bytes=None):
        # Run audio and vision processing in parallel
        audio_task = asyncio.to_thread(self.audio.predict, wav_bytes) if wav_bytes else asyncio.sleep(0, result={"probs": None, "transcript": None})
        vision_task = asyncio.to_thread(self.vision.predict_from_bytes, frame_bytes) if frame_bytes else asyncio.sleep(0, result={"probs": None, "has_face": False})

        audio_out, vision_out = await asyncio.gather(audio_task, vision_task)

        text_out = None
        if self.text and audio_out.get("transcript"):
            # Text prediction depends on ASR, so it runs after audio task is complete
            text_out = await asyncio.to_thread(self.text.predict, audio_out["transcript"])

        # Fusion can also be run in a thread to not block the event loop
        fused = await asyncio.to_thread(
            fuse_probabilities,
            audio_out.get("probs"),
            vision_out.get("probs"),
            text_out.get("probs") if text_out else None
        )

        return {
            "audio": audio_out,
            "vision": vision_out,
            "text": text_out,
            "fusion": fused,
            "labels": EMOTION_LABELS
        }

# singleton
engine = MultimodalEmotionEngine()
