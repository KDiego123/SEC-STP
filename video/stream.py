import cv2
import time

class VideoStream:
    def __init__(self, cam_type, source, reconnect_delay=5):
        self.cam_type = cam_type
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.connect()

    def connect(self):
        print(f"[INFO] Conectando a cámara: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara.")
            self.cap = None

    def read(self):
        if self.cap is None:
            print("[WARN] Reintentando conexión...")
            time.sleep(self.reconnect_delay)
            self.connect()
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("[WARN] Frame perdido. Reintentando...")
            self.cap.release()
            self.cap = None
            return None

        return frame

    def release(self):
        if self.cap:
            self.cap.release()
