import cv2
import sqlite3
import easyocr
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
from math import sqrt
import time # Dodano dla timestamp

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
DETECTION_MODEL_PATH = "best_dziala_90.pt"
DETECTION_MODEL_PLATES_PATH = "best_plates.pt"
VIDEO_SOURCE_BOTTOM = 2
VIDEO_SOURCE_TOP = 0
VIDEO_SOURCE_EXIT = 1  # NOWOŚĆ: Kamera wyjazdowa
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
TOTAL_PARKING_SPOTS = len(PARKING_ZONES) # Łączna liczba miejsc parkingowych
PARKING_OVERLAP_THRESHOLD = 0.60 # Procent pokrycia do uznania miejsca za zajęte

# --- Ustawienia rozdzielczości kamery ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- Konfiguracja Entrypoint ---
ENTRYPOINT_ZONE = (1280, 852, 1642, 1016)
OVERLAP_THRESHOLD = 0.80
x1_ep, y1_ep, x2_ep, y2_ep = ENTRYPOINT_ZONE
ENTRY_GATE_LIGHT = (1350, 828, 1406, 843)
x1_engl, y1_engl, x2_engl, y2_engl = ENTRY_GATE_LIGHT

# --- Konfiguracja Exitpoint (dla kamery górnej) ---
EXITPOINT_ZONE = (198, 842, 534, 989) # Ta strefa nadal służy do detekcji pojazdu z góry
x1_exp, y1_exp, x2_exp, y2_exp = EXITPOINT_ZONE
EXIT_GATE_LIGHT = (205, 810, 263, 828)
x1_exgl, y1_exgl, x2_exgl, y2_exgl = EXIT_GATE_LIGHT

# --- NOWOŚĆ: Konfiguracja strefy detekcji tablicy dla kamery wyjazdowej ---
EXIT_PLATE_DETECTION_ZONE = (0, 0, FRAME_WIDTH, FRAME_HEIGHT) # Może być cały kadr lub określony ROI
x1_epdz, y1_epdz, x2_epdz, y2_epdz = EXIT_PLATE_DETECTION_ZONE



# --- NOWOŚĆ: Konfiguracja wczytywania stanu ---
INITIALIZATION_FRAMES = 100  # Liczba klatek, przez które działa ponowne przypisywanie
REASSIGNMENT_DISTANCE_THRESHOLD = 150  # Maksymalna odległość w pikselach do ponownego przypisania

# --- NOWOŚĆ: Konfiguracja wczytywania stanu ---
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
cap_exit = cv2.VideoCapture(VIDEO_SOURCE_EXIT) # NOWOŚĆ: Otwórz kamerę wyjazdową

cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_exit.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH) # NOWOŚĆ: Ustaw rozdzielczość kamery wyjazdowej
cap_exit.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT) # NOWOŚĆ: Ustaw rozdzielczość kamery wyjazdowej

if not cap_bot.isOpened() or not cap_top.isOpened() or not cap_exit.isOpened(): # NOWOŚĆ: Sprawdź wszystkie kamery
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

# Tworzenie nowej tabeli entries_exits
cursor.execute('''
CREATE TABLE IF NOT EXISTS entries_exits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP
)
''')

# Tworzenie nowej tabeli allowed_plates
cursor.execute('''
CREATE TABLE IF NOT EXISTS allowed_plates (
    plate_number TEXT PRIMARY KEY
)
''')

# Tworzenie nowej tabeli forbidden_moves
cursor.execute('''
CREATE TABLE IF NOT EXISTS forbidden_moves (
    plate_number TEXT PRIMARY KEY,
    forbidden_time TIMESTAMP,
    type TEXT
)
''')

conn.commit()
print("Tabela 'plates' jest gotowa.")

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
        "INSERT INTO allowed_plates (plate_number) VALUES (?)",
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

def calculate_overlap(box1, box2):
    """Calculates the Intersection over Area (IoA) of box1 with box2."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]) # Niepotrzebne dla IoA

    return intersection_area / box1_area if box1_area > 0 else 0


# --- Bufor tablic i danych śledzenia ---
track_to_plate = {}
track_entered_zone = {} # True if vehicle entered entry zone, for one-time OCR
track_history = {}
track_last_y = {}
frame_num = 0

# --- NOWOŚĆ: Globalna zmienna do przechowywania wczytanych pojazdów ---
known_vehicles_from_db = []

# add_allowed_plate_to_db("8008")
# add_allowed_plate_to_db("7007")

# --- Główna pętla programu ---
while True:
    ret_b, frame_b = cap_bot.read()
    ret_t, frame_t = cap_top.read()
    ret_e, frame_e = cap_exit.read() # NOWOŚĆ: Wczytaj klatkę z kamery wyjazdowej

    if not ret_b or not ret_t or not ret_e: # NOWOŚĆ: Sprawdź wszystkie klatki
        print("Nie można odczytać klatki z jednej z kamer. Kończenie programu.")
        break
    frame_num += 1
    
    # --- NOWOŚĆ: Okresowe wczytywanie stanu z bazy danych ---
    if frame_num % DB_RELOAD_INTERVAL_FRAMES == 0:
        print(f"--- Odświeżanie danych z bazy w klatce {frame_num} ---")
        known_vehicles_from_db = load_vehicles_from_db()

    # Detekcja i przygotowanie danych dla trackera (dla kamery górnej)
    rects_for_tracker = []
    results_t = detector(frame_t, imgsz=640, verbose=False)[0]
    for r in results_t.boxes:
        if float(r.conf[0]) < CONFIDENCE_THRESHOLD: continue
        if detector.names[int(r.cls[0])] == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            rects_for_tracker.append((x1, y1, x2, y2))

    tracked_objects = tracker.update(rects_for_tracker)

    # Inicjalizuj zbiór zajętych miejsc parkingowych w tej klatce
    occupied_parking_zones = set()

    # Iteracja po śledzonych obiektach
    for tid, box in tracked_objects.items():
        l, t, r_, b = map(int, box)
        cx, cy = (l + r_) // 2, (t + b) // 2
        vehicle_box = (l, t, r_, b)

        # --- Logika ponownego przypisywania na podstawie stanu z DB ---
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

        # --- Logika kierunku i trajektorii ---
        direction = "S"
        if tid in track_last_y:
            if cy < track_last_y[tid] - 5: direction = "F"
            elif cy > track_last_y[tid] + 5: direction = "B"
        track_last_y[tid] = cy

        if tid not in track_history: track_history[tid] = []
        track_history[tid].append((cx, cy))
        track_history[tid] = track_history[tid][-50:]

        # --- Logika strefy wjazdowej i OCR (z kamery dolnej) ---
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
                            cv2.imshow("Tablica wjazdowa", crop_rgb) # Zmieniona nazwa okna

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
                    
        # --- Logika strefy wyjazdowej (detekcja pojazdu z kamery górnej) i OCR (z nowej kamery wyjazdowej) ---
        overlap_ratio_exit = calculate_overlap(vehicle_box, EXITPOINT_ZONE)

        if overlap_ratio_exit >= OVERLAP_THRESHOLD and tid in track_to_plate:
            plate_to_exit = track_to_plate[tid]
            if is_plate_in_db(plate_to_exit): 
                print(f"Pojazd ID:{tid} wjechał w strefę wyjazdową. Rozpoczynam odczyt OCR z kamery wyjazdowej.")
                
                # NOWOŚĆ: Detekcja tablicy na klatce z kamery wyjazdowej
                results_e = plate_detector(frame_e, imgsz=640, verbose=False)[0]
                found_plate_this_frame_exit = None 

                for r_e in results_e.boxes:
                    if float(r_e.conf[0]) < PLATE_CONFIDENCE_THRESHOLD: continue
                    if plate_detector.names[int(r_e.cls[0])] != PLATE_TARGET_CLASS: continue

                    x1_e, y1_e, x2_e, y2_e = map(int, r_e.xyxy[0])
                    # Opcjonalnie: Przytnij klatkę z kamery wyjazdowej do EXIT_PLATE_DETECTION_ZONE
                    # crop_e_zone = frame_e[y1_epdz:y2_epdz, x1_epdz:x2_epdz] 

                    # Sprawdź, czy wykryta tablica jest w ramach zdefiniowanej strefy dla kamery wyjazdowej (jeśli nie jest całym kadrem)
                    # if x1_e >= x1_epdz and y1_e >= y1_epdz and x2_e <= x2_epdz and y2_e <= y2_epdz:
                    if y1_e < y2_e and x1_e < x2_e: # Upewnij się, że bounding box jest prawidłowy
                        crop = frame_e[y1_e:y2_e, x1_e:x2_e]
                        if crop.size > 0:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            result = ocr.readtext(crop_rgb)
                            cv2.imshow("Tablica wyjazdowa (kamera wyj.)", crop_rgb) # Zmieniona nazwa okna

                            for (bbox, text, conf) in result:
                                cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                                if 4 <= len(cleaned_text) <= 8:
                                    if cleaned_text == plate_to_exit: 
                                        found_plate_this_frame_exit = cleaned_text
                                        print(f"OCR (EasyOCR) przy wyjeździe znalazł tablicę '{found_plate_this_frame_exit}' dla ID:{tid}.")
                                        break
                                    else:
                                        print(f"Odczytana tablica '{cleaned_text}' różni się od przypisanej '{plate_to_exit}'.")

                if found_plate_this_frame_exit:
                    delete_plate_from_db(found_plate_this_frame_exit)
                    update_exit(found_plate_this_frame_exit) 
                    print(f"Usunięto tablicę '{found_plate_this_frame_exit}' z bazy 'plates' (wyjazd).")
                    if tid in track_to_plate: 
                        del track_to_plate[tid]
                    if tid in track_entered_zone: 
                        del track_entered_zone[tid]
                else:
                    print(f"Nie udało się odczytać poprawnej tablicy przy wyjeździe dla ID:{tid} lub tablica nie zgadza się z przypisaną.")


        # --- Sprawdzanie zajętości miejsc parkingowych ---
        for zone_name, zone_box in PARKING_ZONES.items():
            if calculate_overlap(vehicle_box, zone_box) >= PARKING_OVERLAP_THRESHOLD:
                occupied_parking_zones.add(zone_name)
                # Możesz opcjonalnie wyświetlić, która strefa jest zajęta przez który pojazd
                # print(f"Pojazd ID:{tid} zajmuje strefę {zone_name}")


        # Aktualizacja pozycji w bazie, jeśli tablica jest znana
        if tid in track_to_plate:
            plate = track_to_plate[tid]
            update_plate_position(plate, l, t, r_, b)

        # --- Rysowanie na klatce (górnej) ---
        label_text = track_to_plate.get(tid, f"ID:{tid}")
        color = (0, 255, 0) if tid in track_to_plate else (0, 0, 255)
        cv2.rectangle(frame_t, (l, t), (r_, b), color, 2)
        cv2.putText(frame_t, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if tid in track_history:
            pts = track_history[tid]
            for i in range(1, len(pts)):
                cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2)
                
    # --- Rysowanie stref parkingowych (na klatce górnej) ---
    for zone_name, (x1, y1, x2, y2) in PARKING_ZONES.items():
        color = (0, 255, 255) # Domyślny kolor (żółty)
        if zone_name in occupied_parking_zones:
            color = (0, 0, 255) # Zmieniamy na czerwony, jeśli zajęte
        cv2.rectangle(frame_t, (x1, y1), (x2, y2), color, 2) 
        cv2.putText(frame_t, zone_name.replace("ZONE_", "Strefa "), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Rysowanie strefy wjazdu i wyjazdu (na klatce górnej)
    cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2)
    cv2.putText(frame_t, "Strefa Wjazdu", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.rectangle(frame_t, (x1_exp, y1_exp), (x2_exp, y2_exp), (255, 0, 0), 2)
    cv2.putText(frame_t, "Strefa Wyjazdu (dla trackingu)", (x1_exp, y1_exp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # NOWOŚĆ: Rysowanie strefy detekcji tablicy na kamerze wyjazdowej (jeśli inna niż cały kadr)
    # cv2.rectangle(frame_e, (x1_epdz, y1_epdz), (x2_epdz, y2_epdz), (0, 255, 0), 2)
    # cv2.putText(frame_e, "Strefa detekcji tablicy", (x1_epdz, y1_epdz - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Wyświetlanie informacji o fazie inicjalizacji
    if frame_num < INITIALIZATION_FRAMES:
        init_text = f"Faza Inicjalizacji: {frame_num}/{INITIALIZATION_FRAMES}"
        cv2.putText(frame_t, init_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # --- Kolor światła wjazdowego ---
    entry_light_color = (0, 0, 255)  # domyślnie czerwony

    # Sprawdź, czy jakikolwiek dozwolony pojazd znajduje się w strefie ENTRYPOINT
    any_allowed_in_entry = False
    for tid_check, box_check in tracked_objects.items():
        if tid_check in track_to_plate:
            plate_check = track_to_plate[tid_check]
            if is_alowed_plate_in_db(plate_check):
                overlap_ratio_check = calculate_overlap(box_check, ENTRYPOINT_ZONE)
                if overlap_ratio_check >= OVERLAP_THRESHOLD:
                    any_allowed_in_entry = True
                    break

    # Jeśli jest pojazd dozwolony w strefie — zielone światło, inaczej czerwone
    if any_allowed_in_entry:
        entry_light_color = (0, 255, 0)

    # Rysuj prostokąt ENTRY_GATE_LIGHT
    cv2.rectangle(frame_t, (x1_engl, y1_engl), (x2_engl, y2_engl), entry_light_color, -1)

    # --- Kolor światła wyjazdowego ---
    exit_light_color = (0, 0, 255) # domyślnie czerwony

    # Sprawdź, czy jakikolwiek dozwolony pojazd (tj. z tablicą w bazie 'plates')
    # znajduje się w strefie EXITPOINT
    any_allowed_in_exit = False
    for tid_check, box_check in tracked_objects.items():
        if tid_check in track_to_plate:
            plate_check = track_to_plate[tid_check]
            if is_plate_in_db(plate_check): # Sprawdza czy auto jest na parkingu
                overlap_ratio_check = calculate_overlap(box_check, EXITPOINT_ZONE)
                if overlap_ratio_check >= OVERLAP_THRESHOLD:
                    exit_light_color = (0, 255, 0) # Zmieniamy na zielone jeśli pojazd, który jest na parkingu, wjechał w strefę wyjazdu
                    break

    # Rysuj prostokąt EXIT_GATE_LIGHT
    cv2.rectangle(frame_t, (x1_exgl, y1_exgl), (x2_exgl, y2_exgl), exit_light_color, -1)

    # --- Wyświetlanie informacji o zajętości miejsc parkingowych ---
    occupied_count = len(occupied_parking_zones)
    parking_status_text = f"Zajęte: {occupied_count}/{TOTAL_PARKING_SPOTS}"
    cv2.putText(frame_t, parking_status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) # Biały kolor, większa czcionka


    cv2.imshow("Dolna kamera - OCR Wjazd", cv2.resize(frame_b, None, fx=0.5, fy=0.5))
    cv2.imshow("Górna kamera - Tracking Parking", frame_t)
    cv2.imshow("Kamera wyjazdowa - OCR Wyjazd", cv2.resize(frame_e, None, fx=0.5, fy=0.5)) # NOWOŚĆ: Wyświetl kamerę wyjazdową
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sprzątanie
cap_bot.release()
cap_top.release()
cap_exit.release() # NOWOŚĆ: Zwolnij kamerę wyjazdową
conn.close()
cv2.destroyAllWindows()
print("Program zakończony.")