import cv2
import sqlite3
import easyocr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Konfiguracja ---
DETECTION_MODEL_PATH = "best_dziala_90.pt"
VIDEO_SOURCE_BOTTOM = 0  # Dolna kamera – odczyt tablic
VIDEO_SOURCE_TOP = 1     # Górna kamera – śledzenie
TARGET_CLASS = "car"
CONFIDENCE_THRESHOLD = 0.65 # Zwiększona pewność detekcji

# --- Konfiguracja Entrypoint ---
ENTRYPOINT_ZONE = (1260, 800, 1630, 960) # Współrzędne do dostosowania!
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



# --- Bufor tablic i danych śledzenia ---
track_to_plate = {} # przypisanie track_id → tablica
track_entered_zone = {} # track_id → bool (czy pojazd wjechał w strefę)
track_history = {} # historia pozycji dla rysowania trajektorii
track_last_y = {} # ostatnia pozycja Y dla określenia kierunku

frame_num = 0

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
        # DeepSort oczekuje formatu (x, y, width, height) dla detekcji
        detections.append(([x1, y1, x2-x1, y2-y1], float(r.conf[0]), cls))

    tracks = tracker.update_tracks(detections, frame=frame_t)
    for tr in tracks:
        # Pamiętaj, żeby używać tylko potwierdzonych tracków
        if not tr.is_confirmed():
            continue

        tid = tr.track_id
        # Używamy bezpośrednio bounding boxa z DeepSort, jest stabilniejszy
        l, t, r_, b = map(int, tr.to_ltrb()) # lewa, górna, prawa, dolna (bounding box pojazdu)

        # Obliczanie środka boxa dla trajektorii
        cx, cy = (l + r_) // 2, (t + b) // 2

        # --- Logika kierunku ruchu i trajektorii (przeniesiona z drugiego kodu) ---
        direction = "S" # Stojący (Stationary)
        if tid in track_last_y:
            prev_y = track_last_y[tid]
            if cy < prev_y - 5: # Pojazd porusza się w górę obrazu
                direction = "F" # Do przodu (Forward)
            elif cy > prev_y + 5: # Pojazd porusza się w dół obrazu
                direction = "B" # Do tyłu (Backward)
        track_last_y[tid] = cy # Aktualizuj ostatnią pozycję Y

        # Aktualizacja trajektorii ruchu
        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append((cx, cy))
        if len(track_history[tid]) > 50:  # ogranicz długość historii trajektorii
            track_history[tid] = track_history[tid][-50:]

        # --- Logika OCR i Entrypoint (pozostała bez zmian, ale rysowanie oparte na tr.to_ltrb()) ---
        # Sprawdź, czy cały prostokąt pojazdu znajduje się w strefie entrypoint
        is_fully_in_zone = (l >= x1_ep and t >= y1_ep and r_ <= x2_ep and b <= y2_ep)

        # Sprawdź, czy pojazd wjechał w strefę entrypoint (tylko raz)
        if not track_entered_zone.get(tid, False) and is_fully_in_zone:
            track_entered_zone[tid] = True # Oznacz, że pojazd wjechał
            print(f"Pojazd ID:{tid} wjechał CAŁY w strefę entrypoint.")

        # Jeśli pojazd CAŁKOWICIE wjechał w strefę i nie ma jeszcze przypisanej tablicy, spróbuj odczytać OCR
        if track_entered_zone.get(tid, False) and tid not in track_to_plate:
            print(f"Pojazd ID:{tid} w strefie (cały), próba odczytu tablicy z dolnej kamery.")
            # Wykonaj detekcję samochodów na dolnej kamerze
            # Możesz spróbować imgsz=1280 dla lepszej detekcji na dolnej kamerze, jeśli potrzebujesz
            results_b = detector(frame_b, imgsz=640)[0]
            found_plate_this_frame = None

            for r_b in results_b.boxes:
                if float(r_b.conf[0]) < CONFIDENCE_THRESHOLD: continue
                cls_b = detector.names[int(r_b.cls[0])]
                if cls_b != TARGET_CLASS: continue

                x1_b, y1_b, x2_b, y2_b = map(int, r_b.xyxy[0])

                # Upewnij się, że crop jest prawidłowy i nie wychodzi poza obraz
                if y1_b >= 0 and y2_b <= frame_b.shape[0] and x1_b >= 0 and x2_b <= frame_b.shape[1]:
                    crop = frame_b[y1_b:y2_b, x1_b:x2_b]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        ocr_res = ocr.readtext(crop)
                        for bbox, text, conf in ocr_res:
                            # Przetwarzanie i walidacja tablicy
                            txt = text.replace(" ", "").upper()
                            # Dodano bardziej rygorystyczną walidację dla polskich tablic
                            if 4 <= len(txt) <= 10 and txt.isalnum(): # Przykładowa walidacja alfanumeryczna
                                # Opcjonalnie: Dodaj Regex do walidacji PL tablic
                                # import re
                                # if re.fullmatch(r'[A-Z]{2,3}\d{4,5}[A-Z]{0,2}', txt):
                                found_plate_this_frame = txt
                                print(f"Znaleziono tablicę '{found_plate_this_frame}' dla ID:{tid}.")
                                break # Znaleziono sensowną tablicę w tym boxie
                        if found_plate_this_frame:
                            break # Znaleziono sensowną tablicę w tej klatce, nie szukaj dalej

            if found_plate_this_frame:
                track_to_plate[tid] = found_plate_this_frame
                add_plate_to_db(found_plate_this_frame)
            else:
                print(f"Nie znaleziono tablicy dla ID:{tid} w tej klatce. Próbuję dalej...")


        # --- Wyświetl etykietę i bounding box (używamy l, t, r_, b z tracka) ---
        label_text = track_to_plate.get(tid, f"ID:{tid}")
        
        # Dodajemy kierunek ruchu do etykiety, jeśli jest znany
        if direction != "S":
            label_text += f" {direction}"

        color = (0,255,0) if is_plate_in_db(track_to_plate.get(tid, '')) else (0,0,255) # Zielony jeśli w bazie, czerwony jeśli nie

        cv2.rectangle(frame_t, (l,t),(r_,b), color, 2)
        cv2.putText(frame_t, label_text, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Rysowanie trajektorii
        pts = track_history[tid]
        for i in range(1, len(pts)):
            cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2)


    # Narysuj strefę entrypoint na górnej kamerze (dla wizualizacji)
    cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2) # Niebieski prostokąt
    cv2.putText(frame_t, "Entrypoint Zone", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    # Pokaż podgląd
    # cv2.namedWindow("Dolna kamera – OCR", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Dolna kamera – OCR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Dolna kamera – OCR", frame_b)

    # cv2.namedWindow("Górna kamera – tracking i Entrypoint", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Górna kamera – tracking i Entrypoint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Górna kamera – tracking i Entrypoint", frame_t)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sprzątanie
cap_bot.release()
cap_top.release()
conn.close()
cv2.destroyAllWindows()
print("Program zakończony.")