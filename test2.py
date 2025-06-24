import cv2
import sqlite3
import easyocr
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
from math import sqrt
import time

# --- Konfiguracja Tesseract ---
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Zakomentowane, ponieważ w środowisku Canvas może nie być potrzebne lub ścieżka będzie inna

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
DETECTION_MODEL_PATH = "best_dziala_najlepiej.pt"
DETECTION_MODEL_PLATES_PATH = "best_plates.pt"
VIDEO_SOURCE_BOTTOM = 2
VIDEO_SOURCE_TOP = 0
VIDEO_SOURCE_EXIT = 1
TARGET_CLASS = "car"
PLATE_TARGET_CLASS = "plate"
CONFIDENCE_THRESHOLD = 0.50
PLATE_CONFIDENCE_THRESHOLD = 0.75

# --- Konfiguracja miejsc parkingowych ---
PARKING_ZONES = {
    "ZONE_1": (1340, 676, 1624, 810),
    "ZONE_2": (1335, 531, 1610, 664),
    "ZONE_3": (1325, 386, 1595, 519),
    "ZONE_4": (1077, 83, 1215, 321),
    "ZONE_5": (934, 78, 1067, 323),
    "ZONE_6": (797, 76, 922, 321),
    "ZONE_7": (658, 77, 774, 316),
    "ZONE_8": (258, 379, 527, 505),
    "ZONE_9": (235, 524, 509, 650),
    "ZONE_10": (224, 671, 499, 803)
}
TOTAL_PARKING_SPOTS = len(PARKING_ZONES)
PARKING_OVERLAP_THRESHOLD = 0.2

ROAD_ZONES = {
    "ROAD_1": (995, 849, 1231, 1029),
    "ROAD_2": (994, 359, 1289, 845),
    "ROAD_3": (550, 358, 994, 537),
    "ROAD_4": (550, 538, 833, 832),
    "ROAD_5": (602, 835, 831, 1028)
}

ROAD_OVERLAP_THRESHOLD = 0.90
PARKED_OVERLAP_THRESHOLD = 0.50

# --- Ustawienia rozdzielczości kamery ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- Konfiguracja Entrypoint ---
ENTRYPOINT_ZONE = (1280, 852, 1642, 1016)
OVERLAP_THRESHOLD = 0.80
x1_ep, y1_ep, x2_ep, y2_ep = ENTRYPOINT_ZONE
ENTRY_GATE_LIGHT = (1230, 845, 1248, 1023)
x1_engl, y1_engl, x2_engl, y2_engl = ENTRY_GATE_LIGHT

# --- Konfiguracja Exitpoint (dla kamery górnej) ---
EXITPOINT_ZONE = (198, 842, 534, 989)
x1_exp, y1_exp, x2_exp, y2_exp = EXITPOINT_ZONE
EXIT_GATE_LIGHT = (188, 825, 199, 993)
x1_exgl, y1_exgl, x2_exgl, y2_exgl = EXIT_GATE_LIGHT

# --- Konfiguracja strefy detekcji tablicy dla kamery wyjazdowej ---
EXIT_PLATE_DETECTION_ZONE = (0, 0, FRAME_WIDTH, FRAME_HEIGHT)
x1_epdz, y1_epdz, x2_epdz, y2_epdz = EXIT_PLATE_DETECTION_ZONE

# --- Próg pokrycia dla kolizji ---
COLLISION_OVERLAP_THRESHOLD = 0.10

# --- Konfiguracja wczytywania stanu ---
INITIALIZATION_FRAMES = 100
REASSIGNMENT_DISTANCE_THRESHOLD = 200

# --- Konfiguracja odświeżania bazy ---
DB_RELOAD_INTERVAL_FRAMES = 30

# --- Inicjalizacja ---
detector = YOLO(DETECTION_MODEL_PATH)
plate_detector = YOLO(DETECTION_MODEL_PLATES_PATH)
ocr = easyocr.Reader(['pl'])
tracker = CentroidTracker(max_disappeared=50)

# --- Otwórz kamery ---
cap_bot = cv2.VideoCapture(VIDEO_SOURCE_BOTTOM)
cap_top = cv2.VideoCapture(VIDEO_SOURCE_TOP)
cap_exit = cv2.VideoCapture(VIDEO_SOURCE_EXIT)

cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_exit.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_exit.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap_bot.isOpened() or not cap_top.isOpened() or not cap_exit.isOpened():
    print("Błąd: Nie można otworzyć jednej z kamer.")
    exit()

# --- Baza SQLite ---
conn = sqlite3.connect('parking.db')
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
cursor.execute('''
CREATE TABLE IF NOT EXISTS entries_exits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS allowed_plates (
    plate_number TEXT PRIMARY KEY
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS forbidden_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    forbidden_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL
)
''')
conn.commit()
print("Tabele w bazie danych są gotowe.")

def add_entry(plate):
    cursor.execute(
        "INSERT INTO entries_exits (plate_number, entry_time) VALUES (?, CURRENT_TIMESTAMP)",
        (plate,)
    )
    conn.commit()
    print(f"Samochód: '{plate}' wjechał na parking.")

def update_exit(plate):
    cursor.execute(
        "UPDATE entries_exits SET exit_time = CURRENT_TIMESTAMP WHERE plate_number = ? AND exit_time IS NULL",
        (plate,)
    )
    conn.commit()
    print(f"Samochód: '{plate}' wyjechał z parkingu.")

def add_allowed_plate_to_db(plate):
    cursor.execute(
        "INSERT OR IGNORE INTO allowed_plates (plate_number) VALUES (?)",
        (plate,)
    )
    conn.commit()
    print(f"Dodano dozwoloną tablicę '{plate}' do bazy danych.")

def is_alowed_plate_in_db(plate):
    cursor.execute("SELECT 1 FROM allowed_plates WHERE plate_number = ?", (plate,))
    return cursor.fetchone() is not None

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

def delete_plate_from_db(plate):
    if is_plate_in_db(plate):
        cursor.execute(
            "DELETE FROM plates WHERE plate_number = ?",
            (plate,)
        )
        conn.commit()
        print(f"Usunięto tablicę '{plate}' z bazy danych.")

def update_plate_position(plate, x1, y1, x2, y2):
    cursor.execute('''
        UPDATE plates SET x1 = ?, y1 = ?, x2 = ?, y2 = ?, last_update = CURRENT_TIMESTAMP
        WHERE plate_number = ?
    ''', (x1, y1, x2, y2, plate))
    conn.commit()

def add_forbidden_moves(plate, type):
    cursor.execute(
        "INSERT INTO forbidden_moves (plate_number, forbidden_time, type) VALUES (?, CURRENT_TIMESTAMP, ?)",
        (plate, type))
    conn.commit()

def load_vehicles_from_db():
    cursor.execute("SELECT plate_number, x1, y1, x2, y2 FROM plates")
    known_vehicles = []
    for row in cursor.fetchall():
        plate, x1, y1, x2, y2 = row
        if all(v is not None for v in [x1, y1, x2, y2]):
            centroid = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
            known_vehicles.append({'plate': plate, 'centroid': centroid})
    print(f"Wczytano {len(known_vehicles)} znanych pojazdów z bazy danych.")
    return known_vehicles

def calculate_overlap(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return intersection_area / box1_area if box1_area > 0 else 0

# --- Bufory i zmienne stanu ---
track_to_plate = {}
track_entered_zone = {}
track_history = {}
track_last_y = {}
frame_num = 0
known_vehicles_from_db = []
is_currently_offending = {}
vehicle_status = {}

# --- Konfiguracja i zmienne timera bramki wjazdowej ---
GATE_TIMEOUT = 10
gate_opened_time = None

# --- Główna pętla programu ---
while True:
    ret_b, frame_b = cap_bot.read()
    ret_t, frame_t = cap_top.read()
    ret_e, frame_e = cap_exit.read()

    if not ret_b or not ret_t or not ret_e:
        print("Błąd: Nie można odczytać klatki z jednej z kamer. Kończenie programu.")
        break
    frame_num += 1

    if frame_num % DB_RELOAD_INTERVAL_FRAMES == 0:
        print(f"--- Odświeżanie danych z bazy w klatce {frame_num} ---")
        known_vehicles_from_db = load_vehicles_from_db()

    rects_for_tracker = []
    results_t = detector(frame_t, imgsz=640, verbose=False)[0]
    for r in results_t.boxes:
        if float(r.conf[0]) < CONFIDENCE_THRESHOLD: continue
        if detector.names[int(r.cls[0])] == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            rects_for_tracker.append((x1, y1, x2, y2))

    tracked_objects = tracker.update(rects_for_tracker)
    occupied_parking_zones = set()
    collisions_this_frame = {}

    for tid, box in tracked_objects.items():
        l, t, r_, b = map(int, box)
        cx, cy = (l + r_) // 2, (t + b) // 2
        vehicle_box = (l, t, r_, b)

        if tid not in track_to_plate and known_vehicles_from_db:
            current_centroid = (cx, cy)
            min_dist = float('inf')
            best_match_index = -1
            for i, known_vehicle in enumerate(list(known_vehicles_from_db)):
                dist = sqrt((current_centroid[0] - known_vehicle['centroid'][0])**2 + (current_centroid[1] - known_vehicle['centroid'][1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match_index = i
            if min_dist < REASSIGNMENT_DISTANCE_THRESHOLD and best_match_index != -1:
                matched_vehicle = known_vehicles_from_db.pop(best_match_index)
                plate = matched_vehicle['plate']
                track_to_plate[tid] = plate
                print(f"--- Ponowne przypisanie: Obiekt ID:{tid} to tablica '{plate}' (odległość: {min_dist:.0f}px) ---")

        if tid not in track_history: track_history[tid] = []
        track_history[tid].append((cx, cy))
        track_history[tid] = track_history[tid][-50:]

        overlap_ratio_entry = calculate_overlap(vehicle_box, ENTRYPOINT_ZONE)
        if not track_entered_zone.get(tid, False) and overlap_ratio_entry >= OVERLAP_THRESHOLD:
            track_entered_zone[tid] = True
            if tid not in track_to_plate:
                print(f"Pojazd ID:{tid} wjechał w strefę. Rozpoczynam odczyt OCR z kamery dolnej.")
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
                            cv2.imshow("Tablica wjazdowa", crop_rgb)
                            for (bbox, text, conf) in result:
                                cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                                if 4 <= len(cleaned_text) <= 8:
                                    if is_alowed_plate_in_db(cleaned_text):
                                        found_plate_this_frame = cleaned_text
                                        print(f"OCR (EasyOCR) znalazł tablicę '{found_plate_this_frame}' dla ID:{tid}.")
                                        add_entry(found_plate_this_frame)
                                        break
                                    else:
                                        print(f"Brak tablicy: {cleaned_text} w bazie dozwolonych tablic.")
                if found_plate_this_frame:
                    track_to_plate[tid] = found_plate_this_frame
                    add_plate_to_db(found_plate_this_frame, l, t, r_, b)
                else:
                    print(f"Nie udało się odczytać tablicy dla ID:{tid} z kamery dolnej.")
        
        overlap_ratio_exit = calculate_overlap(vehicle_box, EXITPOINT_ZONE)
        if overlap_ratio_exit >= OVERLAP_THRESHOLD and tid in track_to_plate:
            plate_to_exit = track_to_plate[tid]
            if is_plate_in_db(plate_to_exit):
                print(f"Pojazd ID:{tid} w strefie wyjazdowej. Rozpoczynam OCR z kamery wyjazdowej.")
                results_e = plate_detector(frame_e, imgsz=640, verbose=False)[0]
                found_plate_this_frame_exit = None
                for r_e in results_e.boxes:
                    if float(r_e.conf[0]) < PLATE_CONFIDENCE_THRESHOLD: continue
                    if plate_detector.names[int(r_e.cls[0])] != PLATE_TARGET_CLASS: continue
                    x1_e, y1_e, x2_e, y2_e = map(int, r_e.xyxy[0])
                    if y1_e < y2_e and x1_e < x2_e:
                        crop = frame_e[y1_e:y2_e, x1_e:x2_e]
                        if crop.size > 0:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            result = ocr.readtext(crop_rgb)
                            cv2.imshow("Tablica wyjazdowa (kamera wyj.)", crop_rgb)
                            for (bbox, text, conf) in result:
                                cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                                if 4 <= len(cleaned_text) <= 8:
                                    if cleaned_text == plate_to_exit:
                                        found_plate_this_frame_exit = cleaned_text
                                        print(f"OCR przy wyjeździe potwierdził tablicę '{found_plate_this_frame_exit}' dla ID:{tid}.")
                                        break
                                    else:
                                        print(f"Odczytana tablica '{cleaned_text}' różni się od przypisanej '{plate_to_exit}'.")
                if found_plate_this_frame_exit:
                    delete_plate_from_db(found_plate_this_frame_exit)
                    update_exit(found_plate_this_frame_exit)
                    if tid in track_to_plate:
                        if tid in is_currently_offending: del is_currently_offending[tid]
                        if tid in vehicle_status: del vehicle_status[tid]
                        del track_to_plate[tid]
                    if tid in track_entered_zone: del track_entered_zone[tid]
                else:
                    print(f"Nie udało się potwierdzić tablicy przy wyjeździe dla ID:{tid}.")
        
        occupied_zones_by_vehicle = set()
        count = 0
        for zone_name, zone_box in PARKING_ZONES.items():
            if calculate_overlap(vehicle_box, zone_box) >= PARKING_OVERLAP_THRESHOLD:
                occupied_parking_zones.add(zone_name)
                occupied_zones_by_vehicle.add(zone_name)
                count += 1
        
        offense_type_parking = "Zajecie kilku miejsc"
        if count >= 2:
            if tid in track_to_plate:
                plate = track_to_plate[tid]
                if tid not in is_currently_offending or is_currently_offending[tid] != offense_type_parking:
                    print(f"Samochód {plate} (ID:{tid}) ZACZĄŁ zajmować dwa miejsca. Zapisuję wykroczenie.")
                    add_forbidden_moves(plate, offense_type_parking)
                    is_currently_offending[tid] = offense_type_parking
            else:
                print(f"Niezidentyfikowany samochód (ID:{tid}) zajmuje dwa miejsca.")
        else:
            if tid in is_currently_offending and is_currently_offending[tid] == offense_type_parking:
                print(f"Samochód (ID:{tid}) PRZESTAŁ zajmować dwa miejsca.")
                del is_currently_offending[tid]

        offense_type_collision = "Kolizja"
        colliding_with_ids = []
        for other_tid, other_box in tracked_objects.items():
            if tid >= other_tid: continue
            overlap1_2 = calculate_overlap(vehicle_box, other_box)
            overlap2_1 = calculate_overlap(other_box, vehicle_box)
            if overlap1_2 >= COLLISION_OVERLAP_THRESHOLD or overlap2_1 >= COLLISION_OVERLAP_THRESHOLD:
                colliding_with_ids.append(other_tid)
                collision_pair = tuple(sorted((tid, other_tid)))
                collisions_this_frame[collision_pair] = True
        
        if colliding_with_ids:
            if tid in track_to_plate:
                plate = track_to_plate[tid]
                if tid not in is_currently_offending or is_currently_offending[tid] != offense_type_collision:
                    print(f"Samochód {plate} (ID:{tid}) ZACZĄŁ kolizję z ID:{colliding_with_ids}. Zapisuję wykroczenie.")
                    add_forbidden_moves(plate, offense_type_collision)
                    is_currently_offending[tid] = offense_type_collision
            else:
                print(f"Niezidentyfikowany samochód (ID:{tid}) ZACZĄŁ kolizję z ID:{colliding_with_ids}.")
        else:
            if tid in is_currently_offending and is_currently_offending[tid] == offense_type_collision:
                print(f"Samochód (ID:{tid}) PRZESTAŁ kolizję.")
                del is_currently_offending[tid]

        current_vehicle_status = None
        is_parked = any(calculate_overlap(vehicle_box, zone_box) >= PARKED_OVERLAP_THRESHOLD for zone_box in PARKING_ZONES.values())
        if is_parked:
            current_vehicle_status = "zaparkowany"
        else:
            is_on_road = any(calculate_overlap(vehicle_box, zone_box) >= ROAD_OVERLAP_THRESHOLD for zone_box in ROAD_ZONES.values())
            if is_on_road:
                current_vehicle_status = "parkuje"
            else:
                current_vehicle_status = "poza strefami"

        if vehicle_status.get(tid) != current_vehicle_status:
            plate_info = track_to_plate.get(tid, f"ID:{tid}")
            print(f"Samochód {plate_info} zmienił status na: {current_vehicle_status}")
            vehicle_status[tid] = current_vehicle_status

        if tid in track_to_plate:
            plate = track_to_plate[tid]
            update_plate_position(plate, l, t, r_, b)

        label_text = track_to_plate.get(tid, f"ID:{tid}")
        status = vehicle_status.get(tid)
        color = (0, 255, 0) if tid in track_to_plate else (255, 0, 0)
        if status == "zaparkowany": color = (255, 255, 0)
        elif status == "parkuje": color = (0, 165, 255)
        
        if tid in is_currently_offending and is_currently_offending[tid] == offense_type_collision:
            color = (0, 0, 255)
            cv2.putText(frame_t, "KOLIZJA", (l, t + (b-t)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.rectangle(frame_t, (l, t), (r_, b), color, 2)
        cv2.putText(frame_t, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if status:
            cv2.putText(frame_t, status, (l, b + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if tid in track_history:
            pts = track_history[tid]
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None: continue
                cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2)

    for zone_name, (x1, y1, x2, y2) in PARKING_ZONES.items():
        color = (0, 0, 255) if zone_name in occupied_parking_zones else (0, 255, 255)
        cv2.rectangle(frame_t, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_t, zone_name.replace("ZONE_", "Strefa "), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for zone_name, (x1, y1, x2, y2) in ROAD_ZONES.items():
        cv2.rectangle(frame_t, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame_t, zone_name.replace("ROAD_", "Droga "), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2)
    cv2.putText(frame_t, "Strefa Wjazdu", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.rectangle(frame_t, (x1_exp, y1_exp), (x2_exp, y2_exp), (255, 0, 0), 2)
    cv2.putText(frame_t, "Strefa Wyjazdu (tracking)", (x1_exp, y1_exp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if frame_num < INITIALIZATION_FRAMES:
        init_text = f"Faza Inicjalizacji: {frame_num}/{INITIALIZATION_FRAMES}"
        cv2.putText(frame_t, init_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    any_allowed_in_entry = False
    for tid_check, box_check in tracked_objects.items():
        if tid_check in track_to_plate:
            plate_check = track_to_plate[tid_check]
            if is_alowed_plate_in_db(plate_check):
                if calculate_overlap(box_check, ENTRYPOINT_ZONE) >= OVERLAP_THRESHOLD:
                    any_allowed_in_entry = True
                    break
    
    current_occupied_spots = len(occupied_parking_zones)
    can_enter = any_allowed_in_entry and current_occupied_spots < TOTAL_PARKING_SPOTS

    if can_enter:
        if gate_opened_time is None:
            gate_opened_time = time.time()
            print(f"Bramka wjazdowa otwarta. Start timera ({GATE_TIMEOUT}s).")
            entry_light_color = (0, 255, 0)
        else:
            if time.time() - gate_opened_time > GATE_TIMEOUT:
                print(f"Minął czas {GATE_TIMEOUT}s. Bramka wjazdowa zamknięta (timeout).")
                entry_light_color = (0, 0, 255)
            else:
                entry_light_color = (0, 255, 0)
    else:
        if gate_opened_time is not None:
            print("Warunki wjazdu niespełnione. Bramka zamknięta, timer zresetowany.")
        gate_opened_time = None
        entry_light_color = (0, 0, 255)

    cv2.rectangle(frame_t, (x1_engl, y1_engl), (x2_engl, y2_engl), entry_light_color, -1)

    exit_light_color = (0, 0, 255)
    for tid_check, box_check in tracked_objects.items():
        if tid_check in track_to_plate:
            if is_plate_in_db(track_to_plate[tid_check]):
                if calculate_overlap(box_check, EXITPOINT_ZONE) >= OVERLAP_THRESHOLD:
                    exit_light_color = (0, 255, 0)
                    break
    cv2.rectangle(frame_t, (x1_exgl, y1_exgl), (x2_exgl, y2_exgl), exit_light_color, -1)

    parking_status_text = f"Zajete: {current_occupied_spots}/{TOTAL_PARKING_SPOTS}"
    cv2.putText(frame_t, parking_status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Dolna kamera - OCR Wjazd", cv2.resize(frame_b, None, fx=0.5, fy=0.5))
    cv2.imshow("Górna kamera - Tracking Parking", frame_t)
    cv2.imshow("Kamera wyjazdowa - OCR Wyjazd", cv2.resize(frame_e, None, fx=0.5, fy=0.5))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_bot.release()
cap_top.release()
cap_exit.release()
conn.close()
cv2.destroyAllWindows()
print("Program zakończony.")