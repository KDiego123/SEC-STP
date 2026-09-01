from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, frame):
        faces = self.app.get(frame)
        if len(faces) == 0:
            return None
        return faces[0].embedding
