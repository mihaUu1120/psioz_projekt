import cv2
import requests
import time
import numpy as np
from ultralytics import YOLO
import easyocr
import re

# --- Konfiguracja ---
SERVER_URL = "http://127.0.0.1:5000"
VIDEO_SOURCE = 1 # 1 dla kamery na żywo, lub ścieżka do pliku wideo
CONFIDENCE_THRESHOLD_CAR = 0.4
CONFIDENCE_THRESHOLD_PLATE = 0.5

# --- ŚCIEŻKI DO MODELI ---
CAR_MODEL_PATH = "yolov8n.pt"  # Model ogólny do wykrywania aut
# !!! WAŻNE: Podaj tutaj ścieżkę do swojego modelu wykrywającego tablice rejestracyjne !!!
PLATE_MODEL_PATH = "best_plates.pt"

# --- DEFINICJE STREF ---
# Współrzędne wielokątów muszą być dopasowane do Twojego ujęcia z kamery
# Strefa, w której aktywnie szukamy aut i odczytujemy tablice
READ_ZONE_POLY = np.array([[34, 576], [1070, 576], [1070, 1782], [34, 1782]], np.int32)
# Strefa za wirtualną bramką. Służy do weryfikacji, czy auto przejechało
PASSAGE_ZONE_POLY = np.array([[140, 1700], [1018, 1700], [1018, 1902], [140, 1902]], np.int32)

# --- Funkcja pomocnicza do czyszczenia tekstu tablicy ---
def clean_plate_text(text):
    """Usuwa niechciane znaki i próbuje poprawić typowe błędy OCR."""
    # Usuwa spacje i znaki specjalne, zostawia tylko litery i cyfry
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Przykładowe zamiany (można rozbudować)
    text = text.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')
    return text

# --- Główna funkcja programu ---
def main():
    print("Ładowanie modeli...")
    try:
        car_model = YOLO(CAR_MODEL_PATH)
        plate_model = YOLO(PLATE_MODEL_PATH)
        reader = easyocr.Reader(['pl'])
        print("Modele załadowane pomyślnie.")
    except Exception as e:
        print(f"Błąd podczas ładowania modeli: {e}")
        print("Upewnij się, że podałeś poprawną ścieżkę do modelu w PLATE_MODEL_PATH.")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Błąd: Nie można otworzyć źródła wideo: {VIDEO_SOURCE}")
        return

    # Inicjalizacja maszyny stanów
    current_state = "CZEKANIE_NA_AUTO"
    # current_state = "ODCZYTYWANIE_TABLICY"
    gate_status = "ZAMKNIETA"
    message = ""
    last_api_call_time = 0
    state_timer = 0
    last_known_plate = ""
    car_was_in_passage = False

    print("Uruchomiono kamerę wjazdową. Naciśnij 'q', aby zakończyć.")
    print(f"Aktualny stan: {current_state}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Logika maszyny stanów ---

        if current_state == "CZEKANIE_NA_AUTO":
            gate_status = "ZAMKNIETA"
            message = ""
            car_detections = car_model(frame, classes=[2], verbose=False)[0]
            
            car_in_zone = False
            for box in car_detections.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD_CAR:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Sprawdź, czy środek auta jest w strefie odczytu
                    car_center_x = (x1 + x2) // 2
                    if cv2.pointPolygonTest(READ_ZONE_POLY, (car_center_x, y1), False) >= 0:
                        car_in_zone = True
                        
                        detected_car = frame[y1:y2, x1:x2]  # Wytnij prostokąt z obrazu
                        if detected_car.size > 0:  # Sprawdź, czy obraz nie jest pusty
                            cv2.imshow("Wykryty samochod", detected_car)
                        
                        break
            
            if car_in_zone:
                current_state = "ODCZYTYWANIE_TABLICY"
                state_timer = time.time()
                print(f"Nowy stan: {current_state}")

        elif current_state == "ODCZYTYWANIE_TABLICY":
            message = "Odczytywanie tablicy..."
            # Szukaj tablicy tylko w obszarze strefy odczytu
            plate_detections = plate_model(frame, verbose=False)[0]
            plate_found = False

            for box in plate_detections.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD_PLATE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_img = frame[y1:y2, x1:x2]
                    cv2.imshow("Wykryta tablica", plate_img)
                    
                    plate_found = True
                    last_known_plate = "EL8U902"
                    # try:
                    #     ocr_result = reader.readtext(plate_img)
                    #     if ocr_result:
                    #         plate_text = clean_plate_text(ocr_result[0][1])
                    #         # Sprawdź, czy tablica ma sensowną długość
                    #         if 4 < len(plate_text) < 9:
                    #             print(f"Odczytano tablicę: {plate_text}")
                    #             last_known_plate = plate_text
                    #             plate_found = True
                    #             break
                    # except Exception as e:
                    #     print(f"Błąd OCR: {e}")

            if plate_found and time.time() - last_api_call_time > 5: # Unikaj wielokrotnych zapytań
                print(f"Wysyłanie prośby o wjazd dla: {last_known_plate}")
                try:
                    response = requests.post(f"{SERVER_URL}/entry_request", json={"license_plate": last_known_plate}, timeout=3)
                    if response.status_code == 200 and response.json().get("decision") == "OPEN":
                        current_state = "BRAMKA_OTWARTA"
                        state_timer = time.time()
                        print(f"Nowy stan: {current_state}")
                    else:
                        current_state = "ODMOWA_WJAZDU"
                        state_timer = time.time()
                        print(f"Nowy stan: {current_state}. Powód: {response.json().get('reason', 'Nieznany')}")

                except requests.exceptions.RequestException as e:
                    print(f"Błąd połączenia z serwerem: {e}")
                    message = "Blad serwera"
                    time.sleep(2) # Chwila przerwy przed ponowną próbą
                
                last_api_call_time = time.time()
            
            # Timeout, jeśli nie uda się odczytać tablicy
            if time.time() - state_timer > 10:
                current_state = "CZEKANIE_NA_AUTO"
                print(f"Timeout. Nowy stan: {current_state}")


        elif current_state == "BRAMKA_OTWARTA":
            gate_status = "OTWARTA"
            message = f"Witaj {last_known_plate}"

            car_detections = car_model(frame, classes=[2], verbose=False)[0]
            car_in_passage_now = False
            car_in_read_zone_now = False

            for box in car_detections.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD_CAR:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    car_center_y = (y1+y2)//2
                    if cv2.pointPolygonTest(PASSAGE_ZONE_POLY, (x1, car_center_y), False) >= 0:
                        car_in_passage_now = True
                    if cv2.pointPolygonTest(READ_ZONE_POLY, (x1, y2), False) >= 0:
                        car_in_read_zone_now = True
            
            # Sprawdzenie przejazdu
            if car_was_in_passage and not car_in_passage_now:
                print("Pojazd przejechal. Zamykanie bramki.")
                current_state = "CZEKANIE_NA_AUTO"
                print(f"Nowy stan: {current_state}")

            car_was_in_passage = car_in_passage_now

            # Sprawdzenie tailgatingu
            if car_in_read_zone_now:
                message = "ALARM: KOLEJNE AUTO PRZY BRAMCE!"

            # Timeout (kierowca się wycofał)
            if time.time() - state_timer > 20:
                print("Timeout. Auto nie przejechało. Zamykanie bramki.")
                current_state = "CZEKANIE_NA_AUTO"
                print(f"Nowy stan: {current_state}")

        elif current_state == "ODMOWA_WJAZDU":
            gate_status = "ZAMKNIETA"
            message = "ODMOWA WJAZDU"
            if time.time() - state_timer > 5:
                current_state = "CZEKANIE_NA_AUTO"
                print(f"Nowy stan: {current_state}")

        # --- Wizualizacja ---
        viz_frame = frame.copy()
        # Rysowanie stref
        cv2.polylines(viz_frame, [READ_ZONE_POLY], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(viz_frame, "Strefa Odczytu", (READ_ZONE_POLY[0][0], READ_ZONE_POLY[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.polylines(viz_frame, [PASSAGE_ZONE_POLY], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(viz_frame, "Strefa Przejazdu", (PASSAGE_ZONE_POLY[0][0], PASSAGE_ZONE_POLY[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Wyświetlanie statusu i komunikatów
        cv2.putText(viz_frame, f"Stan: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        gate_color = (0, 255, 0) if gate_status == "OTWARTA" else (0, 0, 255)
        cv2.putText(viz_frame, f"Bramka: {gate_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gate_color, 2)
        cv2.putText(viz_frame, message, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


        display_scale = 0.5  # Możesz dostosować np. 0.3, 0.7, 0.8
        viz_frame_resized = cv2.resize(viz_frame, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Kamera Wjazdowa - Cyber Parking", viz_frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()