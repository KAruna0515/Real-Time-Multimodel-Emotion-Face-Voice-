from typing import Optional, List, Dict
import numpy as np

def fuse_probabilities(audio_probs: Optional[List[float]], vision_probs: Optional[List[float]], text_probs: Optional[List[float]], weights=(0.4,0.3,0.3)) -> Dict[str, List[float]]:
    wa,wv,wt = weights
    fused = None
    total = 0.0
    def add(p,w):
        nonlocal fused, total
        if p is None: return
        arr = np.array(p, dtype="float32")
        if fused is None: fused = w*arr
        else: fused += w*arr
        total += w
    add(audio_probs, wa); add(vision_probs, wv); add(text_probs, wt)
    if fused is None or total == 0: return {"fused_probs": None}
    fused = fused / total
    return {"fused_probs": fused.tolist()}
