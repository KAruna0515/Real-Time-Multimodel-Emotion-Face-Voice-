from typing import Dict, Any, List
from .config import get_device, TEXT_EMO_MODEL
import torch

class TextEmotionPipeline:
    def __init__(self):
        self.device = get_device()
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_EMO_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(TEXT_EMO_MODEL).to(self.device)
            self.model.eval()
            self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        except Exception as e:
            raise RuntimeError(f"Failed to load text model: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"probs": None, "labels": None}
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        return {"probs": probs, "labels": self.labels}
