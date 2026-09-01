# utils/body_db.py
import os
import numpy as np

class BodyDB:
    def __init__(self, path="data/bodies"):
        self.db = {}
        self.load(path)

    def load(self, path):
        if not os.path.exists(path):
            return

        for person in os.listdir(path):
            person_dir = os.path.join(path, person)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []
            for f in os.listdir(person_dir):
                emb = np.load(os.path.join(person_dir, f))
                embeddings.append(emb)

            if embeddings:
                mean = np.mean(embeddings, axis=0)
                self.db[person] = mean / np.linalg.norm(mean)

        print(f"[BodyDB] Loaded {len(self.db)} identities")

    def compare(self, emb, thresh=0.55):
        best_name = "Unknown"
        best_score = 0.0

        for name, ref in self.db.items():
            score = np.dot(emb, ref)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < thresh:
            return "Unknown", best_score

        return best_name, best_score
