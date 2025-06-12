import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Konfiguracja ===
#MODEL_PATH = "dataset/runs/detect/train/weights/best.pt"  # ścieżka do wytrenowanego modelu
MODEL_PATH = "best_dziala_90.pt"
VIDEO_SOURCE = 1  # lub np. 0 albo "video.mp4"
TARGET_CLASS = "car"
CONFIDENCE_THRESHOLD = 0.65

# === Inicjalizacja ===
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

track_history = {}
track_last_y = {} 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640)[0]
    detections = []

    # === Przetwarzanie (filtrowanie na podstawie CONFIDENT_THRESHOLD) detekcji ===
    for r in results.boxes:
        conf = float(r.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        
        cls_id = int(r.cls[0])
        class_name = model.names[cls_id]
        if class_name == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, class_name))

    # === Śledzenie ===
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        
        max_difference = 40
        # Szukamy najbardziej pasującej detekcji dla ID
        matched = next((d for d in detections if abs(d[0][0] - l) < max_difference and abs(d[0][1] - t) < max_difference), None)
        if matched:
            x, y, w, h = matched[0]
            cx, cy = x + w // 2, y + h // 2  # środek obiektu

            # Wyznaczanie kierunku ruchu
            direction = "S"
            if track_id in track_last_y:
                prev_y = track_last_y[track_id]
                if cy < prev_y - 5:
                    direction = "F"  # jedzie w górę obrazu
                elif cy > prev_y + 5:
                    direction = "B"  # jedzie w dół obrazu
            track_last_y[track_id] = cy

            # Aktualizacja trajaketorii ruchu
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 50:  # ogranicz długość
                track_history[track_id] = track_history[track_id][-50:]
            
            # Rysowanie prostokąta i ID
            x, y, w, h = matched[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id} {direction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Rysowanie trajektorii
            pts = track_history[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)


    # cv2.namedWindow("YOLOv8 + DeepSORT", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("YOLOv8 + DeepSORT", 1920, 1080)
    cv2.imshow("YOLOv8 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
