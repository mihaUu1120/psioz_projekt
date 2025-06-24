import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from math import sqrt
import time
import database_function
from database_function import ParkingDatabaseManager
from centroid_tracker import CentroidTracker
# Importujemy wszystkie konfiguracje z nowego pliku config.py
from config import (
    DETECTION_MODEL_PATH, DETECTION_MODEL_PLATES_PATH,
    VIDEO_SOURCE_BOTTOM, VIDEO_SOURCE_TOP, VIDEO_SOURCE_EXIT,
    TARGET_CLASS, PLATE_TARGET_CLASS, CONFIDENCE_THRESHOLD, PLATE_CONFIDENCE_THRESHOLD,
    PARKING_ZONES, TOTAL_PARKING_SPOTS, PARKING_OVERLAP_THRESHOLD,
    ROAD_ZONES, ROAD_OVERLAP_THRESHOLD, PARKED_OVERLAP_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT,
    ENTRYPOINT_ZONE, OVERLAP_THRESHOLD, ENTRY_GATE_LIGHT,
    EXITPOINT_ZONE, EXIT_GATE_LIGHT,
    COLLISION_OVERLAP_THRESHOLD,
    INITIALIZATION_FRAMES, REASSIGNMENT_DISTANCE_THRESHOLD,
    DB_RELOAD_INTERVAL_FRAMES,
    GATE_TIMEOUT
)


# Funkcja do obliczania pokrycia między dwoma ramkami ograniczającymi
def calculate_overlap(box1, box2):
    """
    Oblicza współczynnik pokrycia box1 względem powierzchni box1.

    Args:
        box1 (tuple): Współrzędne (x1, y1, x2, y2) pierwszej ramki ograniczającej.
        box2 (tuple): Współrzędne (x1, y1, x2, y2) drugiej ramki ograniczającej.

    Returns:
        float: Stosunek powierzchni przecięcia do powierzchni box1.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return intersection_area / box1_area if box1_area > 0 else 0

# --- Inicjalizacja ---
detector = YOLO(DETECTION_MODEL_PATH)
plate_detector = YOLO(DETECTION_MODEL_PLATES_PATH)
ocr = easyocr.Reader(['pl']) # Inicjalizacja EasyOCR z językiem polskim
tracker = CentroidTracker(max_disappeared=50) # max_disappeared nadal tutaj, można też przenieść do config
db_manager = ParkingDatabaseManager() # Inicjalizacja menedżera bazy danych

# --- Otwórz kamery ---
cap_bot = cv2.VideoCapture(VIDEO_SOURCE_BOTTOM)
cap_top = cv2.VideoCapture(VIDEO_SOURCE_TOP)
cap_exit = cv2.VideoCapture(VIDEO_SOURCE_EXIT)

# Ustaw rozdzielczości kamer
cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_exit.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_exit.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Sprawdź, czy kamery zostały pomyślnie otwarte
if not cap_bot.isOpened() or not cap_top.isOpened() or not cap_exit.isOpened():
    print("Błąd: Nie można otworzyć jednej z kamer. Kończenie programu.")
    exit()

# Dodaj przykładowe dozwolone tablice dla testów
db_manager.add_allowed_plate("8008")
db_manager.add_allowed_plate("7007")
db_manager.add_allowed_plate("2115")

# --- Bufory i zmienne stanu ---
track_to_plate = {}       # Mapuje identyfikator trackera na wykryty numer tablicy
track_entered_zone = {}   # Śledzi, czy pojazd wjechał do strefy wjazdu
track_history = {}        # Przechowuje historyczne centroidy do rysowania trajektorii
frame_num = 0             # Bieżący numer klatki
known_vehicles_from_db = [] # Bufor pojazdów załadowanych z bazy danych
is_currently_offending = {} # Śledzi pojazdy aktualnie popełniające zabronione ruchy
vehicle_status = {}       # Śledzi bieżący status pojazdów (zaparkowany, parkujący itp.)

# --- Zmienna timera bramy wjazdowej ---
gate_opened_time = None # Sygnatura czasowa otwarcia bramy wjazdowej

# Rozpakowanie współrzędnych strefy wjazdu/wyjazdu
# Te zmienne są używane do rysowania na ramce, a ich wartości pochodzą z config.py
x1_ep, y1_ep, x2_ep, y2_ep = ENTRYPOINT_ZONE
x1_exp, y1_exp, x2_exp, y2_exp = EXITPOINT_ZONE
x1_engl, y1_engl, x2_engl, y2_engl = ENTRY_GATE_LIGHT
x1_exgl, y1_exgl, x2_exgl, y2_exgl = EXIT_GATE_LIGHT

# --- Główna pętla programu ---
while True:
    # Odczytaj klatki ze wszystkich kamer
    ret_b, frame_b = cap_bot.read()
    ret_t, frame_t = cap_top.read()
    ret_e, frame_e = cap_exit.read()

    # Zakończ, jeśli którejś z kamer nie uda się odczytać klatki
    if not ret_b or not ret_t or not ret_e:
        print("Błąd: Nie można odczytać klatki z jednej z kamer. Kończenie programu.")
        break
    frame_num += 1

    # Okresowo ładuj znane pojazdy z bazy danych
    if frame_num % DB_RELOAD_INTERVAL_FRAMES == 0:
        print(f"--- Odświeżanie danych z bazy danych w klatce {frame_num} ---")
        known_vehicles_from_db = db_manager.load_vehicles_from_active_parking()

    # Wykryj samochody w klatce z kamery górnej do śledzenia
    rects_for_tracker = []
    results_t = detector(frame_t, imgsz=640, verbose=False)[0] # Uruchom wnioskowanie YOLO
    for r in results_t.boxes:
        if float(r.conf[0]) < CONFIDENCE_THRESHOLD: continue # Filtruj według pewności
        if detector.names[int(r.cls[0])] == TARGET_CLASS: # Filtruj według klasy (samochód)
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            rects_for_tracker.append((x1, y1, x2, y2))

    # Zaktualizuj tracker centroidów o nowe detekcje
    tracked_objects = tracker.update(rects_for_tracker)
    occupied_parking_zones = set() # Do śledzenia aktualnie zajętych miejsc parkingowych
    collisions_this_frame = {} # Do śledzenia kolizji w bieżącej klatce

    # Przetwórz każdy śledzony obiekt
    for tid, box in tracked_objects.items():
        l, t, r_, b = map(int, box)
        cx, cy = (l + r_) // 2, (t + b) // 2
        vehicle_box = (l, t, r_, b)

        # Próbuj ponownie przypisać znaną tablicę z bazy danych do nowego śladu
        # Pomaga to w przypadku błędów śledzenia lub początkowej konfiguracji
        if tid not in track_to_plate and known_vehicles_from_db:
            current_centroid = (cx, cy)
            min_dist = float('inf')
            best_match_index = -1
            # Znajdź najbliższy znany pojazd z bazy danych
            for i, known_vehicle in enumerate(list(known_vehicles_from_db)): # Użyj list(), aby umożliwić pop
                dist = sqrt((current_centroid[0] - known_vehicle['centroid'][0])**2 +
                            (current_centroid[1] - known_vehicle['centroid'][1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match_index = i
            # Jeśli znaleziono wystarczająco bliskie dopasowanie, przypisz tablicę
            if min_dist < REASSIGNMENT_DISTANCE_THRESHOLD and best_match_index != -1:
                matched_vehicle = known_vehicles_from_db.pop(best_match_index) # Usuń z kandydatów
                plate = matched_vehicle['plate']
                track_to_plate[tid] = plate
                print(f"--- Ponowne przypisanie: Obiekt ID:{tid} to tablica '{plate}' (odległość: {min_dist:.0f}px) ---")

        # Zaktualizuj historię śledzenia dla rysowania trajektorii
        if tid not in track_history: track_history[tid] = []
        track_history[tid].append((cx, cy))
        track_history[tid] = track_history[tid][-50:] # Zachowaj ostatnie 50 centroidów

        # Logika Punktu Wjazdowego (Kamera Dolna - OCR)
        overlap_ratio_entry = calculate_overlap(vehicle_box, ENTRYPOINT_ZONE)
        if not track_entered_zone.get(tid, False) and overlap_ratio_entry >= OVERLAP_THRESHOLD:
            track_entered_zone[tid] = True # Oznacz pojazd jako wjeżdżający do strefy
            if tid not in track_to_plate: # Próbuj OCR tylko wtedy, gdy tablica nie została jeszcze przypisana
                print(f"Pojazd ID:{tid} wjechał w strefę wjazdu. Rozpoczynam odczyt OCR z kamery dolnej.")
                results_b = plate_detector(frame_b, imgsz=640, verbose=False)[0]
                found_plate_this_frame = None
                for r_b in results_b.boxes:
                    if float(r_b.conf[0]) < PLATE_CONFIDENCE_THRESHOLD: continue
                    if plate_detector.names[int(r_b.cls[0])] != PLATE_TARGET_CLASS: continue
                    x1_b, y1_b, x2_b, y2_b = map(int, r_b.xyxy[0])
                    # Upewnij się, że współrzędne kadrowania są prawidłowe
                    if y1_b < y2_b and x1_b < x2_b:
                        crop = frame_b[y1_b:y2_b, x1_b:x2_b]
                        if crop.size > 0: # Upewnij się, że kadrowanie nie jest puste
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            result = ocr.readtext(crop_rgb)
                            cv2.imshow("Entry Plate OCR (Bottom Camera)", crop_rgb) # Pokaż przyciętą tablicę
                            for (bbox, text, conf) in result:
                                cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper())) # Wyczyść tekst tablicy
                                if 4 <= len(cleaned_text) <= 8: # Podstawowa walidacja długości tablicy
                                    if db_manager.is_allowed_plate(cleaned_text):
                                        found_plate_this_frame = cleaned_text
                                        print(f"OCR (EasyOCR) znalazł dozwoloną tablicę '{found_plate_this_frame}' dla ID:{tid}.")
                                        db_manager.add_entry(found_plate_this_frame)
                                        break # Zatrzymaj po znalezieniu jednej prawidłowej tablicy
                                    else:
                                        print(f"Wykryta tablica: {cleaned_text} nie znajduje się w bazie danych dozwolonych tablic.")
                if found_plate_this_frame:
                    track_to_plate[tid] = found_plate_this_frame
                    db_manager.add_plate_to_active_parking(found_plate_this_frame, l, t, r_, b)
                else:
                    print(f"Nie udało się odczytać prawidłowej tablicy dla ID:{tid} z kamery dolnej.")

        # Logika Punktu Wyjazdowego (Kamera Wyjściowa - OCR)
        overlap_ratio_exit = calculate_overlap(vehicle_box, EXITPOINT_ZONE)
        if overlap_ratio_exit >= OVERLAP_THRESHOLD and tid in track_to_plate:
            plate_to_exit = track_to_plate[tid]
            if db_manager.is_plate_in_active_parking(plate_to_exit): # Sprawdź, czy tablica jest zarejestrowana jako zaparkowana
                print(f"Pojazd ID:{tid} w strefie wyjazdu. Rozpoczynam OCR z kamery wyjazdowej.")
                results_e = plate_detector(frame_e, imgsz=640, verbose=False)[0]
                found_plate_this_frame_exit = None
                for r_e in results_e.boxes:
                    if float(r_e.conf[0]) < PLATE_CONFIDENCE_THRESHOLD: continue
                    if plate_detector.names[int(r_e.cls[0])] != PLATE_TARGET_CLASS: continue
                    x1_e, y1_e, x2_e, y2_e = map(int, r_e.xyxy[0])
                    # Upewnij się, że współrzędne kadrowania są prawidłowe
                    if y1_e < y2_e and x1_e < x2_e:
                        crop = frame_e[y1_e:y2_e, x1_e:x2_e]
                        if crop.size > 0: # Upewnij się, że kadrowanie nie jest puste
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            result = ocr.readtext(crop_rgb)
                            cv2.imshow("Exit Plate OCR (Exit Camera)", crop_rgb) # Pokaż przyciętą tablicę
                            for (bbox, text, conf) in result:
                                cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper())) # Wyczyść tekst tablicy
                                if 4 <= len(cleaned_text) <= 8: # Podstawowa walidacja długości tablicy
                                    if cleaned_text == plate_to_exit: # Potwierdź, że tablica pasuje do śledzonej
                                        found_plate_this_frame_exit = cleaned_text
                                        print(f"OCR przy wyjeździe potwierdził tablicę '{found_plate_this_frame_exit}' dla ID:{tid}.")
                                        break # Zatrzymaj po znalezieniu pasującej tablicy
                                    else:
                                        print(f"Odczytana tablica '{cleaned_text}' różni się od przypisanej '{plate_to_exit}'.")
                if found_plate_this_frame_exit:
                    db_manager.delete_plate_from_active_parking(found_plate_this_frame_exit)
                    db_manager.update_exit(found_plate_this_frame_exit)
                    # Wyczyść stan trackera dla tego wyjeżdżającego pojazdu
                    if tid in track_to_plate: del track_to_plate[tid]
                    if tid in is_currently_offending: del is_currently_offending[tid]
                    if tid in vehicle_status: del vehicle_status[tid]
                    if tid in track_entered_zone: del track_entered_zone[tid]
                else:
                    print(f"Nie udało się potwierdzić tablicy przy wyjeździe dla ID:{tid}.")

        # Wykroczenie Parkingowe: Zajmowanie wielu miejsc
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
                    print(f"Samochód {plate} (ID:{tid}) ZACZĄŁ zajmować wiele miejsc. Rejestruję wykroczenie.")
                    db_manager.add_forbidden_move(plate, offense_type_parking)
                    is_currently_offending[tid] = offense_type_parking
            else:
                print(f"Niezidentyfikowany samochód (ID:{tid}) zajmuje wiele miejsc.")
        else:
            if tid in is_currently_offending and is_currently_offending[tid] == offense_type_parking:
                print(f"Samochód (ID:{tid}) PRZESTAŁ zajmować wiele miejsc.")
                del is_currently_offending[tid]

        # Wykrywanie Kolizji
        offense_type_collision = "Kolizja"
        colliding_with_ids = []
        for other_tid, other_box in tracked_objects.items():
            if tid >= other_tid: continue # Unikaj podwójnych sprawdzeń i samokolizji
            overlap1_2 = calculate_overlap(vehicle_box, other_box)
            overlap2_1 = calculate_overlap(other_box, vehicle_box)
            if overlap1_2 >= COLLISION_OVERLAP_THRESHOLD or overlap2_1 >= COLLISION_OVERLAP_THRESHOLD:
                colliding_with_ids.append(other_tid)
                collision_pair = tuple(sorted((tid, other_tid))) # Użyj posortowanej krotki dla unikalnego klucza pary
                collisions_this_frame[collision_pair] = True # Oznacz kolizję dla pary
        
        if colliding_with_ids:
            if tid in track_to_plate:
                plate = track_to_plate[tid]
                if tid not in is_currently_offending or is_currently_offending[tid] != offense_type_collision:
                    print(f"Samochód {plate} (ID:{tid}) ZACZĄŁ kolizję z ID:{colliding_with_ids}. Rejestruję wykroczenie.")
                    db_manager.add_forbidden_move(plate, offense_type_collision)
                    is_currently_offending[tid] = offense_type_collision
            else:
                print(f"Niezidentyfikowany samochód (ID:{tid}) ZACZĄŁ kolizję z ID:{colliding_with_ids}.")
        else:
            if tid in is_currently_offending and is_currently_offending[tid] == offense_type_collision:
                print(f"Samochód (ID:{tid}) PRZESTAŁ kolizję.")
                del is_currently_offending[tid]

        # Określenie statusu pojazdu (zaparkowany, parkujący, poza strefami)
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

        # Zapisz zmianę statusu
        if vehicle_status.get(tid) != current_vehicle_status:
            plate_info = track_to_plate.get(tid, f"ID:{tid}")
            print(f"Samochód {plate_info} zmienił status na: {current_vehicle_status}")
            vehicle_status[tid] = current_vehicle_status

        # Zaktualizuj pozycję tablicy w bazie danych, jeśli jest śledzona
        if tid in track_to_plate:
            plate = track_to_plate[tid]
            db_manager.update_plate_position(plate, l, t, r_, b)

        # Rysowanie ramek ograniczających, etykiet i trajektorii na klatce z kamery górnej
        label_text = track_to_plate.get(tid, f"ID:{tid}")
        status = vehicle_status.get(tid)
        color = (0, 255, 0) if tid in track_to_plate else (255, 0, 0) # Zielony, jeśli tablica przypisana, Czerwony w przeciwnym razie
        if status == "zaparkowany": color = (255, 255, 0) # Żółty dla zaparkowanych
        elif status == "parkuje": color = (0, 165, 255) # Pomarańczowy dla parkujących/na drodze
        
        # Zastąp kolor, jeśli wykryto kolizję
        if tid in is_currently_offending and is_currently_offending[tid] == offense_type_collision:
            color = (0, 0, 255) # Czerwony dla kolizji
            cv2.putText(frame_t, "KOLIZJA", (l, t + (b-t)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.rectangle(frame_t, (l, t), (r_, b), color, 2)
        cv2.putText(frame_t, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if status:
            cv2.putText(frame_t, status, (l, b + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if tid in track_history:
            pts = track_history[tid]
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None: continue
                cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2) # Cyjanowa trajektoria

    # Rysuj strefy parkingowe i strefy drogowe na klatce z kamery górnej
    for zone_name, (x1, y1, x2, y2) in PARKING_ZONES.items():
        color = (0, 0, 255) if zone_name in occupied_parking_zones else (0, 255, 255) # Czerwony, jeśli zajęty, Żółty w przeciwnym razie
        cv2.rectangle(frame_t, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_t, zone_name.replace("ZONE_", "Strefa "), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for zone_name, (x1, y1, x2, y2) in ROAD_ZONES.items():
        cv2.rectangle(frame_t, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta dla dróg
        cv2.putText(frame_t, zone_name.replace("ROAD_", "Droga "), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Rysuj strefy wjazdu/wyjazdu
    cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2) # Niebieski dla wjazdu
    cv2.putText(frame_t, "Strefa Wjazdu", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.rectangle(frame_t, (x1_exp, y1_exp), (x2_exp, y2_exp), (255, 0, 0), 2) # Niebieski dla wyjazdu
    cv2.putText(frame_t, "Strefa Wyjazdu (tracking)", (x1_exp, y1_exp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Wyświetl komunikat o fazie inicjalizacji
    if frame_num < INITIALIZATION_FRAMES:
        init_text = f"Faza Inicjalizacji: {frame_num}/{INITIALIZATION_FRAMES}"
        cv2.putText(frame_t, init_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Logika Światła Bramy Wjazdowej: Zielone, jeśli dozwolona tablica w strefie wjazdu I parking nie jest pełny
    any_allowed_in_entry = False
    for tid_check, box_check in tracked_objects.items():
        if tid_check in track_to_plate:
            plate_check = track_to_plate[tid_check]
            if db_manager.is_allowed_plate(plate_check):
                if calculate_overlap(box_check, ENTRYPOINT_ZONE) >= OVERLAP_THRESHOLD:
                    any_allowed_in_entry = True
                    break # Znaleziono dozwolony samochód w strefie wjazdu

    current_occupied_spots = len(occupied_parking_zones)
    can_enter = any_allowed_in_entry and current_occupied_spots < TOTAL_PARKING_SPOTS

    if can_enter:
        if gate_opened_time is None:
            gate_opened_time = time.time()
            print(f"Brama wjazdowa otwarta. Start timera ({GATE_TIMEOUT}s).")
            entry_light_color = (0, 255, 0) # Zielone światło
        else:
            if time.time() - gate_opened_time > GATE_TIMEOUT:
                print(f"Limit czasu ({GATE_TIMEOUT}s) przekroczony. Brama wjazdowa zamknięta (timeout).")
                entry_light_color = (0, 0, 255) # Czerwone światło
            else:
                entry_light_color = (0, 255, 0) # Nadal zielone
    else:
        if gate_opened_time is not None:
            print("Warunki wjazdu nie spełnione. Brama zamknięta, timer zresetowany.")
        gate_opened_time = None
        entry_light_color = (0, 0, 255) # Czerwone światło

    cv2.rectangle(frame_t, (x1_engl, y1_engl), (x2_engl, y2_engl), entry_light_color, -1) # Narysuj światło bramy wjazdowej

    # Logika Światła Bramy Wyjazdowej: Zielone, jeśli śledzony samochód z aktywnym rekordem parkingowym znajduje się w strefie wyjazdu
    exit_light_color = (0, 0, 255) # Domyślnie czerwone
    for tid_check, box_check in tracked_objects.items():
        if tid_check in track_to_plate:
            # Sprawdź, czy tablica jest aktywnie zaparkowana (tj. w tabeli 'plates')
            if db_manager.is_plate_in_active_parking(track_to_plate[tid_check]):
                if calculate_overlap(box_check, EXITPOINT_ZONE) >= OVERLAP_THRESHOLD:
                    exit_light_color = (0, 255, 0) # Zielone światło
                    break
    cv2.rectangle(frame_t, (x1_exgl, y1_exgl), (x2_exgl, y2_exgl), exit_light_color, -1) # Narysuj światło bramy wyjazdowej

    # Wyświetl status parkowania
    parking_status_text = f"Zajęte: {current_occupied_spots}/{TOTAL_PARKING_SPOTS}"
    cv2.putText(frame_t, parking_status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Wyświetl klatki
    cv2.imshow("Dolna kamera - OCR Wjazd", cv2.resize(frame_b, None, fx=0.5, fy=0.5))
    cv2.imshow("Górna kamera - Śledzenie Parkingu", frame_t)
    cv2.imshow("Kamera wyjazdowa - OCR Wyjazd", cv2.resize(frame_e, None, fx=0.5, fy=0.5))

    # Wyjdź po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby kamer i zamknij połączenie z bazą danych
cap_bot.release()
cap_top.release()
cap_exit.release()
db_manager.close() # Zamknij połączenie z bazą danych
cv2.destroyAllWindows()
print("Program zakończony.")
