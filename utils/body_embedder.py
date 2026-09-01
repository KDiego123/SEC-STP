# utils/body_embedder.py
import torch
import torchreid
import numpy as np
import cv2

class BodyEmbedder:
    def __init__(self):
        self.device = "cpu"
        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)

    def get_embedding(self, img_bgr):
        try:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 128))  # estándar ReID
            img = img.transpose(2, 0, 1) / 255.0

            tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                feat = self.model(tensor)

            emb = feat.cpu().numpy().flatten()
            return emb / np.linalg.norm(emb)

        except Exception:
            return None
