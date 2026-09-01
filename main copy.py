import cv2
import time

from utils.config_loader import load_config
from video.stream import VideoStream

def main():
    config = load_config()

    cam_cfg = config["camera"]
    proc_cfg = config["processing"]
    disp_cfg = config["display"]

    stream = VideoStream(
        cam_type=cam_cfg["type"],
        source=cam_cfg["source"],
        reconnect_delay=cam_cfg["reconnect_delay"]
    )

    target_fps = proc_cfg["target_fps"]
    delay = 1.0 / target_fps

    print("[INFO] Sistema iniciado. Presiona 'q' para salir.")

    while True:
        start = time.time()

        frame = stream.read()
        if frame is None:
            continue

        if proc_cfg["resize"] != 1.0:
            frame = cv2.resize(
                frame, None,
                fx=proc_cfg["resize"],
                fy=proc_cfg["resize"]
            )

        if disp_cfg["show"]:
            cv2.imshow(disp_cfg["window_name"], frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed = time.time() - start
        if elapsed < delay:
            time.sleep(delay - elapsed)

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
