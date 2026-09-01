import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition

from utils.config_loader import load_config
from video.stream import VideoStream
from utils.face_embedder import FaceEmbedder


# -------------------------
# Filtro humano (anti-gato)
# -------------------------
def has_human_face(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb, model="hog")
    return len(faces) > 0


# -------------------------
# Distancia coseno (ArcFace)
# -------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    config = load_config()

    # -------------------------
    # Inicializaciones
    # -------------------------
    model = YOLO(config["model"]["yolo_model"])
    embedder = FaceEmbedder()

    cap = VideoStream(
        cam_type=config["camera"]["type"],
        source=config["camera"]["source"],
        reconnect_delay=config["camera"]["reconnect_delay"]
    )

    # -------------------------
    # Base de datos simple (MVP)
    # -------------------------
    known_embeddings = []
    known_names = []

    print("Sistema iniciado. Presiona 'q' para salir.")
    print("Presiona 'r' para registrar una nueva persona.")

    while True:
        frame = cap.read()
        if frame is None:
            continue

        results = model(frame, conf=config["model"]["confidence_threshold"])

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Solo personas
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                # -------------------------
                # Filtro humano
                # -------------------------
                if not has_human_face(person_crop):
                    continue

                # -------------------------
                # Embedding ArcFace
                # -------------------------
                embedding = embedder.get_embedding(person_crop)
                if embedding is None:
                    continue

                name = "DESCONOCIDO"
                color = (0, 0, 255)

                if known_embeddings:
                    sims = [cosine_similarity(embedding, e) for e in known_embeddings]
                    best_idx = int(np.argmax(sims))
                    best_score = sims[best_idx]

                    # Umbral típico ArcFace
                    if best_score > 0.5:
                        name = f"{known_names[best_idx]} ({best_score:.2f})"
                        color = (0, 255, 0)

                # -------------------------
                # Dibujo
                # -------------------------
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        cv2.imshow("Face MVP - Reconocimiento Facial", frame)

        key = cv2.waitKey(1) & 0xFF

        # Registrar nueva persona
        if key == ord("r"):
            if embedding is not None:
                nombre = input("Nombre de la persona: ")
                known_embeddings.append(embedding)
                known_names.append(nombre)
                print(f"[OK] {nombre} registrado.")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
