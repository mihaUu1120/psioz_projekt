import cv2
import sqlite3
import easyocr
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Konfiguracja ---
DETECTION_MODEL_PATH = "best_dziala_90.pt"
VIDEO_SOURCE_BOTTOM = 0  # Dolna kamera – odczyt tablic
VIDEO_SOURCE_TOP = 1     # Górna kamera – śledzenie
TARGET_CLASS = "car"
CONFIDENCE_THRESHOLD = 0.65 # Zwiększona pewność detekcji

# --- Konfiguracja Entrypoint ---
ENTRYPOINT_ZONE = (1242, 845, 1637, 1014) # Współrzędne do dostosowania!
OVERLAP_THRESHOLD = 0.80 # (NOWOŚĆ) Wymagane 80% pokrycia strefy przez pojazd
# Współrzędne (x1_ep, y1_ep, x2_ep, y2_ep) dla ENTRYPOINT_ZONE
x1_ep, y1_ep, x2_ep, y2_ep = ENTRYPOINT_ZONE

# --- Ustawienia rozdzielczości kamery ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- Inicjalizacja ---
detector = YOLO(DETECTION_MODEL_PATH)
ocr = easyocr.Reader(['pl']) # Zmieniono język na polski
tracker = DeepSort(max_age=90, n_init=3) # Zwiększono max_age i dodano n_init

# --- Otwórz kamery ---
cap_bot = cv2.VideoCapture(VIDEO_SOURCE_BOTTOM)
cap_top = cv2.VideoCapture(VIDEO_SOURCE_TOP)

# Ustaw rozdzielczość dla obu kamer
cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Sprawdzenie, czy kamery zostały poprawnie otwarte
if not cap_bot.isOpened():
    print(f"Błąd: Nie można otworzyć kamery dolnej (ID: {VIDEO_SOURCE_BOTTOM}). "
          "Upewnij się, że jest podłączona i nie jest używana przez inną aplikację.")
    exit()

if not cap_top.isOpened():
    print(f"Błąd: Nie można otworzyć kamery górnej (ID: {VIDEO_SOURCE_TOP}). "
          "Upewnij się, że jest podłączona i nie jest używana przez inną aplikację.")
    exit()

# Opcjonalnie: Sprawdź, jaka rozdzielczość faktycznie została ustawiona
actual_width_bot = int(cap_bot.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height_bot = int(cap_bot.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Dolna kamera - Rzeczywista rozdzielczość: {actual_width_bot}x{actual_height_bot}")

actual_width_top = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height_top = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Górna kamera - Rzeczywista rozdzielczość: {actual_width_top}x{actual_height_top}")


# --- Baza SQLite ---
conn = sqlite3.connect('plates.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS plates (
        plate_number TEXT PRIMARY KEY
    )
''')
conn.commit()

def is_plate_in_db(plate):
    """Sprawdza, czy tablica jest w bazie danych."""
    cursor.execute("SELECT 1 FROM plates WHERE plate_number = ?", (plate,))
    return cursor.fetchone() is not None

def add_plate_to_db(plate):
    """Dodaje tablicę do bazy danych, jeśli jej tam nie ma."""
    if not is_plate_in_db(plate):
        cursor.execute("INSERT INTO plates (plate_number) VALUES (?)", (plate,))
        conn.commit()
        print(f"Dodano tablicę '{plate}' do bazy danych.")

def calculate_overlap(vehicle_box, zone_box):
    """
    Oblicza stosunek pola przecięcia do pola prostokąta pojazdu.
    Zwraca wartość od 0.0 do 1.0.
    Format pudełka: (x1, y1, x2, y2)
    """
    vx1, vy1, vx2, vy2 = vehicle_box
    zx1, zy1, zx2, zy2 = zone_box

    # Oblicz współrzędne prostokąta będącego częścią wspólną (przecięciem)
    inter_x1 = max(vx1, zx1)
    inter_y1 = max(vy1, zy1)
    inter_x2 = min(vx2, zx2)
    inter_y2 = min(vy2, zy2)

    # Oblicz pole części wspólnej
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    # Oblicz pole prostokąta pojazdu
    vehicle_area = (vx2 - vx1) * (vy2 - vy1)

    # Unikaj dzielenia przez zero, jeśli pole pojazdu jest równe 0
    if vehicle_area == 0:
        return 0.0

    # Oblicz i zwróć stosunek pokrycia
    overlap_ratio = intersection_area / vehicle_area
    return overlap_ratio


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Przygotowuje obraz wyciętej tablicy rejestracyjnej do OCR.
    """
    # 1. Zmiana rozmiaru (interpolacja sześcienna dla lepszej jakości)
    # OCR działa najlepiej, gdy wysokość obrazu to ok. 50-100 pikseli.
    h, w = image.shape[:2]
    if h < 50:
        scale_factor = 100 / h
        width = int(w * scale_factor)
        height = int(h * scale_factor)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    # 2. Konwersja do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Zwiększenie kontrastu (CLAHE - znacznie lepsze niż zwykłe wyrównanie)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 4. Delikatne rozmycie w celu usunięcia szumu (Median Blur dobrze radzi sobie z "pieprzem i solą")
    blurred = cv2.medianBlur(contrast_enhanced, 3)

    # 5. Wyostrzanie (Twoja propozycja) - używamy "jądra" (kernel)
    # Działa dobrze, ale może wzmocnić też szum, dlatego stosujemy po rozmyciu.
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # OPCJONALNIE: Binaryzacja (próg adaptacyjny jest lepszy od stałego)
    # Czasami pomaga, a czasami nie - warto przetestować.
    # binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)
    
    return sharpened # lub 'binary', jeśli zdecydujesz się na ten krok



# --- Bufor tablic i danych śledzenia ---
track_to_plate = {} # przypisanie track_id → tablica
track_entered_zone = {} # track_id → bool (czy pojazd wjechał w strefę)
track_history = {} # historia pozycji dla rysowania trajektorii
track_last_y = {} # ostatnia pozycja Y dla określenia kierunku

frame_num = 0

# --- Główna pętla programu (zaktualizowana) ---
while True:
    ret_b, frame_b = cap_bot.read()
    ret_t, frame_t = cap_top.read()

    if not ret_b or not ret_t:
        print("Problem z kamerą (brak klatek). Upewnij się, że kamery są podłączone i działają.")
        break

    frame_num += 1

    # Górna kamera – detekcja + DeepSORT
    detections = []
    results_t = detector(frame_t, imgsz=640, verbose=False)[0]
    for r in results_t.boxes:
        if float(r.conf[0]) < CONFIDENCE_THRESHOLD: continue
        cls = detector.names[int(r.cls[0])]
        if cls != TARGET_CLASS: continue

        x1, y1, x2, y2 = map(int, r.xyxy[0])
        detections.append(([x1, y1, x2-x1, y2-y1], float(r.conf[0]), cls))

    tracks = tracker.update_tracks(detections, frame=frame_t)
    for tr in tracks:
        if not tr.is_confirmed():
            continue

        tid = tr.track_id
        l, t, r_, b = map(int, tr.to_ltrb())
        cx, cy = (l + r_) // 2, (t + b) // 2

        # --- Logika kierunku ruchu (bez zmian) ---
        direction = "S"
        if tid in track_last_y:
            prev_y = track_last_y[tid]
            if cy < prev_y - 5: direction = "F"
            elif cy > prev_y + 5: direction = "B"
        track_last_y[tid] = cy

        # --- Logika trajektorii (bez zmian) ---
        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append((cx, cy))
        if len(track_history[tid]) > 50:
            track_history[tid] = track_history[tid][-50:]
        
        # --- NOWA LOGIKA WEJŚCIA W STREFĘ ---
        # Zamiast sprawdzać, czy pojazd jest w całości w strefie,
        # obliczamy procent jego pokrycia ze strefą.
        
        vehicle_box = (l, t, r_, b)
        overlap_ratio = calculate_overlap(vehicle_box, ENTRYPOINT_ZONE)
        is_in_zone_by_overlap = overlap_ratio >= OVERLAP_THRESHOLD

        # Sprawdź, czy pojazd wjechał w strefę wystarczająco głęboko (tylko raz)
        if not track_entered_zone.get(tid, False) and is_in_zone_by_overlap:
            track_entered_zone[tid] = True # Oznacz, że pojazd aktywował strefę
            print(f"Pojazd ID:{tid} pokrył {overlap_ratio:.0%} strefy. Rozpoczynam odczyt tablicy.")

        # --- Logika OCR (uruchamiana po aktywacji strefy) ---
        if track_entered_zone.get(tid, False) and tid not in track_to_plate:
            # Użyj verbose=False również tutaj dla spójności
            results_b = detector(frame_b, imgsz=640, verbose=False)[0]
            found_plate_this_frame = None

            for r_b in results_b.boxes:
                if float(r_b.conf[0]) < CONFIDENCE_THRESHOLD: continue
                cls_b = detector.names[int(r_b.cls[0])]
                if cls_b != TARGET_CLASS: continue

                x1_b, y1_b, x2_b, y2_b = map(int, r_b.xyxy[0])
                if y1_b >= 0 and y2_b <= frame_b.shape[0] and x1_b >= 0 and x2_b <= frame_b.shape[1]:
                    crop = frame_b[y1_b:y2_b, x1_b:x2_b]
                    if crop.size > 0:
                        ocr_res = ocr.readtext(crop)
                        for _, text, _ in ocr_res:
                            txt = text.replace(" ", "").upper()
                            if 4 <= len(txt) <= 10 and txt.isalnum():
                                found_plate_this_frame = txt
                                print(f"Znaleziono tablicę '{found_plate_this_frame}' dla ID:{tid}.")
                                break
                    if found_plate_this_frame: break
            
            if found_plate_this_frame:
                track_to_plate[tid] = found_plate_this_frame
                add_plate_to_db(found_plate_this_frame)
            else:
                print(f"Nie znaleziono tablicy dla ID:{tid} w tej klatce.")

        # --- Rysowanie na klatce (bez zmian) ---
        label_text = track_to_plate.get(tid, f"ID:{tid}")
        if direction != "S":
            label_text += f" {direction}"
        
        color = (0, 255, 0) if is_plate_in_db(track_to_plate.get(tid, '')) else (0, 0, 255)
        cv2.rectangle(frame_t, (l, t), (r_, b), color, 2)
        cv2.putText(frame_t, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if tid in track_history:
            pts = track_history[tid]
            for i in range(1, len(pts)):
                cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2)

    # Rysowanie strefy i wyświetlanie (bez zmian)
    cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2)
    cv2.putText(frame_t, "Entrypoint Zone", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imshow("Dolna kamera – OCR", frame_b)
    cv2.imshow("Górna kamera – tracking i Entrypoint", frame_t)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sprzątanie
cap_bot.release()
cap_top.release()
conn.close()
cv2.destroyAllWindows()
print("Program zakończony.")