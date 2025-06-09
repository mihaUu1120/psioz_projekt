import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Konfiguracja ===
MODEL_PATH = "dataset/runs/detect/train/weights/best.pt"  # wytrenowany model
#MODEL_PATH = "best_v2.pt"
VIDEO_SOURCE = 1  # 0 = kamerka, lub podaj ścieżkę np. "video.mp4"
TARGET_CLASS = "car"  # klasa do śledzenia

# === Inicjalizacja ===
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detekcja YOLO
    results = model(frame, imgsz=640, conf=0.4)[0]
    detections = []

    for r in results.boxes:
        cls_id = int(r.cls[0])
        class_name = model.names[cls_id]
        if class_name == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    # Śledzenie
    tracks = tracker.update_tracks(detections, frame=frame)

    # Rysowanie
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
