# utils/face_db.py
import os
import cv2
import numpy as np
from utils.arcface_embedder import ArcFaceEmbedder

class FaceDatabase:
    def __init__(self, faces_dir, device="cpu"):
        self.faces_dir = faces_dir
        self.embedder = ArcFaceEmbedder(device)
        self.db = []

    def load(self):
        for person_name in os.listdir(self.faces_dir):
            person_path = os.path.join(self.faces_dir, person_name)

            if not os.path.isdir(person_path):
                continue

            embeddings = []

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                emb = self.embedder.get_embedding(img)
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                mean_embedding = np.mean(embeddings, axis=0)
                self.db.append({
                    "name": person_name,
                    "embedding": mean_embedding
                })

        print(f"[FaceDB] Loaded {len(self.db)} identities")

    def compare(self, embedding, threshold=0.6):
        best_name = "Unknown"
        best_score = -1

        for person in self.db:
            score = np.dot(embedding, person["embedding"]) / (
                np.linalg.norm(embedding) * np.linalg.norm(person["embedding"])
            )

            if score > best_score:
                best_score = score
                best_name = person["name"]

        if best_score < threshold:
            return "Unknown", best_score

        return best_name, best_score
