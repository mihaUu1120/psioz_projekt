import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
import os
from math import sqrt

# --- Konfiguracja Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Klasa Centroid Tracker (bez zmian) ---
# --- Klasa do śledzenia obiektów (Centroid Tracker) ---
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.boxes = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.boxes

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_boxes = {}
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
            input_boxes[i] = (x1, y1, x2, y2)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_boxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    dist = sqrt((object_centroids[i][0] - input_centroids[j][0])**2 + (object_centroids[i][1] - input_centroids[j][1])**2)
                    D[i, j] = dist

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = input_boxes[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], input_boxes[col])
        return self.boxes

# --- Konfiguracja ---
DETECTION_MODEL_PATH = "best_dziala_90.pt"
DETECTION_MODEL_PLATES_PATH = "best_plates.pt"
VIDEO_SOURCE_BOTTOM = 1
VIDEO_SOURCE_TOP = 0
TARGET_CLASS = "car"
PLATE_TARGET_CLASS = "plate"
CONFIDENCE_THRESHOLD = 0.50
PLATE_CONFIDENCE_THRESHOLD = 0.75

# --- Konfiguracja Entrypoint ---
ENTRYPOINT_ZONE = (1280, 852, 1642, 1016)
OVERLAP_THRESHOLD = 0.80
x1_ep, y1_ep, x2_ep, y2_ep = ENTRYPOINT_ZONE

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
INITIALIZATION_FRAMES = 100

# --- Inicjalizacja ---
detector = YOLO(DETECTION_MODEL_PATH)
plate_detector = YOLO(DETECTION_MODEL_PLATES_PATH)
ocr = easyocr.Reader(['pl'])
tracker = CentroidTracker(max_disappeared=50)

# --- Folder zapisu tablic ---
SAVE_DIR = "zapisane_tablice"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Otwórz kamery ---
cap_bot = cv2.VideoCapture(VIDEO_SOURCE_BOTTOM)
cap_top = cv2.VideoCapture(VIDEO_SOURCE_TOP)
cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap_bot.isOpened() or not cap_top.isOpened():
    print("Błąd: Nie można otworzyć jednej z kamer.")
    exit()

# --- Bufory ---
track_to_plate = {}
track_entered_zone = {}
track_history = {}
track_last_y = {}
frame_num = 0

def calculate_overlap(vehicle_box, zone_box):
    vx1, vy1, vx2, vy2 = vehicle_box
    zx1, zy1, zx2, zy2 = zone_box
    inter_x1 = max(vx1, zx1)
    inter_y1 = max(vy1, zy1)
    inter_x2 = min(vx2, zx2)
    inter_y2 = min(vy2, zy2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height
    vehicle_area = (vx2 - vx1) * (vy2 - vy1)
    return intersection_area / vehicle_area if vehicle_area > 0 else 0

# --- Główna pętla ---
while True:
    ret_b, frame_b = cap_bot.read()
    ret_t, frame_t = cap_top.read()
    if not ret_b or not ret_t: break
    frame_num += 1

    rects_for_tracker = []
    results_t = detector(frame_t, imgsz=640, verbose=False)[0]
    for r in results_t.boxes:
        if float(r.conf[0]) < CONFIDENCE_THRESHOLD: continue
        if detector.names[int(r.cls[0])] == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            rects_for_tracker.append((x1, y1, x2, y2))

    tracked_objects = tracker.update(rects_for_tracker)

    for tid, box in tracked_objects.items():
        l, t, r_, b = map(int, box)
        cx, cy = (l + r_) // 2, (t + b) // 2

        if tid not in track_last_y:
            track_last_y[tid] = cy
        direction = "S"
        if cy < track_last_y[tid] - 5: direction = "F"
        elif cy > track_last_y[tid] + 5: direction = "B"
        track_last_y[tid] = cy

        if tid not in track_history: track_history[tid] = []
        track_history[tid].append((cx, cy))
        track_history[tid] = track_history[tid][-50:]

        vehicle_box = (l, t, r_, b)
        overlap_ratio = calculate_overlap(vehicle_box, ENTRYPOINT_ZONE)

        if not track_entered_zone.get(tid, False) and overlap_ratio >= OVERLAP_THRESHOLD:
            track_entered_zone[tid] = True
            print(f"Pojazd ID:{tid} wjechał w strefę. Rozpoczynam odczyt OCR.")

            results_b = plate_detector(frame_b, imgsz=640, verbose=False)[0]
            found_plate_this_frame = None

            for r_b in results_b.boxes:
                if float(r_b.conf[0]) < PLATE_CONFIDENCE_THRESHOLD: continue
                if plate_detector.names[int(r_b.cls[0])] != PLATE_TARGET_CLASS: continue

                x1_b, y1_b, x2_b, y2_b = map(int, r_b.xyxy[0])
                if y1_b < y2_b and x1_b < x2_b:
                    crop = frame_b[y1_b:y2_b, x1_b:x2_b]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        result = ocr.readtext(crop_rgb)
                        cv2.imshow("Tablica", crop_rgb)

                        for (bbox, text, conf) in result:
                            cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                            if 5 <= len(cleaned_text) <= 8:
                                found_plate_this_frame = cleaned_text
                                print(f"OCR (EasyOCR) znalazł tablicę '{found_plate_this_frame}' dla ID:{tid}.")

                                # Zapisz obraz tablicy
                                filename = f"{SAVE_DIR}/{found_plate_this_frame}_{frame_num}.jpg"
                                cv2.imwrite(filename, crop)
                                print(f"Zapisano tablicę jako: {filename}")
                                break

            if found_plate_this_frame:
                track_to_plate[tid] = found_plate_this_frame
            else:
                print(f"Nie udało się odczytać tablicy dla ID:{tid}.")

        label_text = track_to_plate.get(tid, f"ID:{tid}")
        color = (0, 255, 0) if tid in track_to_plate else (0, 0, 255)
        cv2.rectangle(frame_t, (l, t), (r_, b), color, 2)
        cv2.putText(frame_t, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if tid in track_history:
            pts = track_history[tid]
            for i in range(1, len(pts)):
                cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2)

    cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2)
    cv2.putText(frame_t, "Entrypoint Zone", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if frame_num < INITIALIZATION_FRAMES:
        init_text = f"Faza Inicjalizacji: {frame_num}/{INITIALIZATION_FRAMES}"
        cv2.putText(frame_t, init_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Dolna kamera - OCR", cv2.resize(frame_b, None, fx=0.5, fy=0.5))
    cv2.imshow("Górna kamera - Tracking", frame_t)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_bot.release()
cap_top.release()
cv2.destroyAllWindows()
print("Program zakończony.")
