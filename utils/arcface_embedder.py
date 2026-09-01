# utils/arcface_embedder.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class ArcFaceEmbedder:
    def __init__(self, device="cpu"):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0 if device == "cuda" else -1)

    def get_embedding(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(image_rgb)

        if len(faces) == 0:
            return None

        # tomamos el rostro más grande
        faces = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return faces[0].embedding
