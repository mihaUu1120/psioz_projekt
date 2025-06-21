import cv2
import sqlite3
import easyocr
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
from math import sqrt

# --- Konfiguracja Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

# --- Ustawienia rozdzielczości kamery ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- NOWOŚĆ: Konfiguracja wczytywania stanu ---
INITIALIZATION_FRAMES = 100  # Liczba klatek, przez które działa ponowne przypisywanie
REASSIGNMENT_DISTANCE_THRESHOLD = 150  # Maksymalna odległość w pikselach do ponownego przypisania

# --- NOWOŚĆ: Konfiguracja wczytywania stanu ---
# Ustawienie klatek do inicjalizacji
INITIALIZATION_FRAMES = 100
REASSIGNMENT_DISTANCE_THRESHOLD = 150
# Odświeżanie pozycji pojazdów na parkingu 
DB_RELOAD_INTERVAL_FRAMES = 300

# --- Inicjalizacja ---
detector = YOLO(DETECTION_MODEL_PATH)
plate_detector = YOLO(DETECTION_MODEL_PLATES_PATH)
ocr = easyocr.Reader(['pl'])
tracker = CentroidTracker(max_disappeared=50)

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

# --- Baza SQLite ---
conn = sqlite3.connect('plates.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS plates (
    plate_number TEXT PRIMARY KEY,
    x1 INTEGER,
    y1 INTEGER,
    x2 INTEGER,
    y2 INTEGER,
    last_update TEXT DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()
print("Tabela 'plates' jest gotowa.")

def is_plate_in_db(plate):
    cursor.execute("SELECT 1 FROM plates WHERE plate_number = ?", (plate,))
    return cursor.fetchone() is not None

def add_plate_to_db(plate, x1, y1, x2, y2):
    if not is_plate_in_db(plate):
        cursor.execute(
            "INSERT INTO plates (plate_number, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?)",
            (plate, x1, y1, x2, y2)
        )
        conn.commit()
        print(f"Dodano tablicę '{plate}' do bazy danych.")

def update_plate_position(plate, x1, y1, x2, y2):
    cursor.execute('''
        UPDATE plates SET x1 = ?, y1 = ?, x2 = ?, y2 = ?, last_update = CURRENT_TIMESTAMP
        WHERE plate_number = ?
    ''', (x1, y1, x2, y2, plate))
    conn.commit()

# --- NOWOŚĆ: Funkcja do wczytywania stanu z bazy ---
def load_vehicles_from_db():
    """Wczytuje zapisane pojazdy i ich ostatnie pozycje (jako centroidy) z bazy danych."""
    cursor.execute("SELECT plate_number, x1, y1, x2, y2 FROM plates")
    known_vehicles = []
    for row in cursor.fetchall():
        plate, x1, y1, x2, y2 = row
        if all(v is not None for v in [x1, y1, x2, y2]):
            centroid = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
            known_vehicles.append({'plate': plate, 'centroid': centroid})
    print(f"Wczytano {len(known_vehicles)} znanych pojazdów z bazy danych.")
    return known_vehicles

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

# --- Bufor tablic i danych śledzenia ---
track_to_plate = {}
track_entered_zone = {}
track_history = {}
track_last_y = {}
frame_num = 0

# --- NOWOŚĆ: Globalna zmienna do przechowywania wczytanych pojazdów ---
known_vehicles_from_db = []

# --- Główna pętla programu ---
while True:
    ret_b, frame_b = cap_bot.read()
    ret_t, frame_t = cap_top.read()
    if not ret_b or not ret_t: break
    frame_num += 1
    
    # --- NOWOŚĆ: Okresowe wczytywanie stanu z bazy danych ---
    if frame_num % DB_RELOAD_INTERVAL_FRAMES == 0:
        print(f"--- Odświeżanie danych z bazy w klatce {frame_num} ---")
        known_vehicles_from_db = load_vehicles_from_db()

    # Detekcja i przygotowanie danych dla trackera
    rects_for_tracker = []
    results_t = detector(frame_t, imgsz=640, verbose=False)[0]
    for r in results_t.boxes:
        if float(r.conf[0]) < CONFIDENCE_THRESHOLD: continue
        if detector.names[int(r.cls[0])] == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            rects_for_tracker.append((x1, y1, x2, y2))

    tracked_objects = tracker.update(rects_for_tracker)

    # Iteracja po śledzonych obiektach
    for tid, box in tracked_objects.items():
        l, t, r_, b = map(int, box)
        cx, cy = (l + r_) // 2, (t + b) // 2

        # --- NOWOŚĆ: Logika ponownego przypisywania na podstawie stanu z DB ---
        # This part will now continuously try to re-assign if a plate isn't known
        # and a matching known vehicle is nearby from the reloaded DB data.
        if tid not in track_to_plate and known_vehicles_from_db:
            current_centroid = (cx, cy)
            min_dist = float('inf')
            best_match_index = -1

            # Iterate through a *copy* to allow safe removal if a match is made
            for i, known_vehicle in enumerate(list(known_vehicles_from_db)):
                dist = sqrt((current_centroid[0] - known_vehicle['centroid'][0])**2 + (current_centroid[1] - known_vehicle['centroid'][1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match_index = i

            if min_dist < REASSIGNMENT_DISTANCE_THRESHOLD and best_match_index != -1:
                matched_vehicle = known_vehicles_from_db.pop(best_match_index) # Remove matched vehicle to avoid re-matching
                plate = matched_vehicle['plate']
                track_to_plate[tid] = plate
                print(f"--- Ponowne przypisanie: Obiekt ID:{tid} to tablica '{plate}' (odległość: {min_dist:.0f}px) ---")


        # --- Logika kierunku i trajektorii ---
        direction = "S"
        if tid in track_last_y:
            if cy < track_last_y[tid] - 5: direction = "F"
            elif cy > track_last_y[tid] + 5: direction = "B"
        track_last_y[tid] = cy

        if tid not in track_history: track_history[tid] = []
        track_history[tid].append((cx, cy))
        track_history[tid] = track_history[tid][-50:]

        # --- Logika strefy i OCR ---
        vehicle_box = (l, t, r_, b)
        overlap_ratio = calculate_overlap(vehicle_box, ENTRYPOINT_ZONE)
        
        if not track_entered_zone.get(tid, False) and overlap_ratio >= OVERLAP_THRESHOLD:
            track_entered_zone[tid] = True
            if tid not in track_to_plate: # Odczytuj tylko, jeśli nie ma jeszcze tablicy
                print(f"Pojazd ID:{tid} wjechał w strefę. Rozpoczynam odczyt OCR.")
                
                # Uruchomienie logiki OCR... (reszta kodu bez zmian)
                results_b = plate_detector(frame_b, imgsz=640, verbose=False)[0]
                found_plate_this_frame = None
                for r_b in results_b.boxes:
                    if float(r_b.conf[0]) < PLATE_CONFIDENCE_THRESHOLD: continue
                    if plate_detector.names[int(r_b.cls[0])] != PLATE_TARGET_CLASS: continue

                    x1_b, y1_b, x2_b, y2_b = map(int, r_b.xyxy[0])
                    if y1_b < y2_b and x1_b < x2_b:
                        crop = frame_b[y1_b:y2_b, x1_b:x2_b]
                        if crop.size > 0:
                            # crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            # config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                            # raw_text = pytesseract.image_to_string(crop_rgb, lang='eng+pol', config=config)
                            # cv2.imshow("Tablica", crop_rgb)
                            # combined_text = ''.join(re.findall(r'[A-Z0-9]', raw_text.upper()))
                            
                            # if 5 <= len(combined_text) <= 8:
                            #     found_plate_this_frame = combined_text
                            #     print(f"OCR Znalazł tablicę '{found_plate_this_frame}' dla ID:{tid}.")
                            #     break
                            
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            result = ocr.readtext(crop_rgb)
                            cv2.imshow("Tablica", crop_rgb)

                            for (bbox, text, conf) in result:
                                cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                                if 5 <= len(cleaned_text) <= 8:
                                    found_plate_this_frame = cleaned_text
                                    print(f"OCR (EasyOCR) znalazł tablicę '{found_plate_this_frame}' dla ID:{tid}.")
                                    break

                
                if found_plate_this_frame:
                    track_to_plate[tid] = found_plate_this_frame
                    add_plate_to_db(found_plate_this_frame, l, t, r_, b)
                else:
                    print(f"Nie udało się odczytać tablicy dla ID:{tid}.")


        # Aktualizacja pozycji w bazie, jeśli tablica jest znana
        if tid in track_to_plate:
            plate = track_to_plate[tid]
            update_plate_position(plate, l, t, r_, b)

        # --- Rysowanie na klatce ---
        label_text = track_to_plate.get(tid, f"ID:{tid}")
        color = (0, 255, 0) if tid in track_to_plate else (0, 0, 255)
        cv2.rectangle(frame_t, (l, t), (r_, b), color, 2)
        cv2.putText(frame_t, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if tid in track_history:
            pts = track_history[tid]
            for i in range(1, len(pts)):
                cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2)

    # Rysowanie strefy i wyświetlanie
    cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2)
    cv2.putText(frame_t, "Entrypoint Zone", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Wyświetlanie informacji o fazie inicjalizacji
    if frame_num < INITIALIZATION_FRAMES:
        init_text = f"Faza Inicjalizacji: {frame_num}/{INITIALIZATION_FRAMES}"
        cv2.putText(frame_t, init_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Dolna kamera - OCR", cv2.resize(frame_b, None, fx=0.5, fy=0.5))
    cv2.imshow("Górna kamera - Tracking", frame_t)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sprzątanie
cap_bot.release()
cap_top.release()
conn.close()
cv2.destroyAllWindows()
print("Program zakończony.")