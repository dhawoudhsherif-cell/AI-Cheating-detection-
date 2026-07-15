import cv2 as cv
import os
import time
import threading
from datetime import datetime
from ultralytics import YOLO
from playsound import playsound
from typing import List, Tuple                                                                                                          
# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "yolov8n.pt"
LOGS_DIR = "logs"
SNAPSHOTS_DIR = "snapshots"
RECORDS_DIR = "records"

CONFIDENCE_THRESHOLD = 0.5
FPS = 20.0
EVENT_DURATION_SEC = 5  # seconds

ALERT_SOUNDS = {
    "cell phone": "alert_cell.wav",
    "book": "alert_book.wav",
    "laptop": "alert_laptop.wav"
}

# Ensure directories exist
for folder in [LOGS_DIR, SNAPSHOTS_DIR, RECORDS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ================================
# HELPER FUNCTIONS
# ================================
def initialize_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def initialize_camera() -> Tuple[cv.VideoCapture, int, int]:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height

def log_detection(detected_objects: List[str]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOGS_DIR, "cheating_log.txt")

    with open(log_file, "a") as f:
        f.write(f"{timestamp}: Detected {', '.join(detected_objects)}\n")

    return timestamp

def save_snapshot(frame, timestamp: str) -> None:
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{timestamp}.jpg")
    cv.imwrite(path, frame)

def save_event_video(frames: List, timestamp: str, width: int, height: int) -> None:
    path = os.path.join(RECORDS_DIR, f"event_{timestamp}.avi")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(path, fourcc, FPS, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def play_alert_sound(detected_objects: List[str]) -> None:
    first_obj = detected_objects[0]
    if first_obj in ALERT_SOUNDS:
        try:
            playsound(ALERT_SOUNDS[first_obj])
        except Exception as e:
            print(f"⚠️ Sound error: {e}")

def play_alert_sound_threaded(detected_objects: List[str]) -> None:
    threading.Thread(
        target=play_alert_sound,
        args=(detected_objects,),
        daemon=True
    ).start()

def detect_objects(model: YOLO, frame):
    results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            if label in ALERT_SOUNDS:
                detected_objects.append(label)

    return frame, detected_objects

# ================================
# MAIN FUNCTION
# ================================
def main() -> None:
    print("🚀 Starting Cheating Detection... Press 'q' to quit.")

    model = initialize_model(MODEL_PATH)
    cap, width, height = initialize_camera()

    recording = False
    event_frames = []
    event_start_time = 0
    timestamp = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = detect_objects(model, frame)

        if detected_objects and not recording:
            recording = True
            event_start_time = time.time()
            event_frames = []

            timestamp = log_detection(detected_objects)
            save_snapshot(frame, timestamp)
            play_alert_sound_threaded(detected_objects)

            print(f"🎥 Recording started at {timestamp}")

        if recording:
            event_frames.append(frame.copy())
            if time.time() - event_start_time >= EVENT_DURATION_SEC:
                save_event_video(event_frames, timestamp, width, height)
                print(f"💾 Event video saved: {timestamp}")
                recording = False

        cv.imshow("Cheating Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print("✅ Detection ended.")

# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    main()