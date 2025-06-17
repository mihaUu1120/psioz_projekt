import cv2
import requests
import time
import numpy as np
from ultralytics import YOLO
from math import sqrt
import json

# --- Konfiguracja ---
# Adres URL serwera centralnego
SERVER_URL = "http://127.0.0.1:5000" 
# Źródło wideo: 0 dla kamery na żywo, lub ścieżka do pliku np. "parking_test.mp4"
VIDEO_SOURCE = 0 
# Co ile sekund wysyłać aktualizacje pozycji do serwera
UPDATE_INTERVAL_SECONDS = 5 
# Próg pewności dla detekcji YOLO
CONFIDENCE_THRESHOLD = 0.5 
# Ścieżka do modelu
DETECTION_MODEL_PATH = "best_dziala_90.pt"

# Ustawienia rozdzielczości kamery
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Wielokąt definiujący strefę, w której pojawiają się nowe auta po wjeździe
# Te współrzędne muszą być dopasowane do Twojego ujęcia z kamery
ENTRY_ZONE_POLY = np.array([[1280, 852], [1642, 852], [1642, 1016], [1280, 1016]], np.int32)


# --- Klasa do śledzenia obiektów (Centroid Tracker) ---
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}  # Słownik przechowujący ID: centroid
        self.boxes = {}    # Słownik przechowujący ID: bounding_box
        self.disappeared = {}  # Słownik przechowujący ID: licznik klatek od zniknięcia
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
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_boxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Obliczanie odległości między starymi a nowymi centroidami
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

# --- Główna funkcja programu ---
def main():
    # Załadowanie modelu YOLOv8 (pobierze się automatycznie)
    print("Ładowanie modelu YOLOv8...")
    model = YOLO(DETECTION_MODEL_PATH)
    
    # Inicjalizacja trackera i słownika do asocjacji ID z tablicami
    tracker = CentroidTracker()
    plate_associations = {} # Słownik przechowujący {tracker_id: 'KR12345'}

    # Otwarcie źródła wideo
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Błąd: Nie można otworzyć źródła wideo: {VIDEO_SOURCE}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
    last_update_time = time.time()

    print("Uruchomiono kamerę górną. Naciśnij 'q', aby zakończyć.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detekcja obiektów
        # Szukamy klasy 'car' (id=0)
        results = model(frame, classes=[0], verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2))
        
        # Aktualizacja trackera
        tracked_objects = tracker.update(detections)

        # Logika asocjacji nowych obiektów z tablicami
        for object_id, box in tracked_objects.items():
            if object_id not in plate_associations:
                # Obiekt jest nowy, sprawdzamy czy pojawił się w strefie wjazdu
                center_x = (box[0] + box[2]) // 2
                center_y = (box[1] + box[3]) // 2
                
                if cv2.pointPolygonTest(ENTRY_ZONE_POLY, (center_x, center_y), False) >= 0:
                    print(f"Nowy obiekt ID: {object_id} w strefie wjazdu. Próba powiązania z tablicą...")
                    # UWAGA: W realnym systemie tutaj powinien być dedykowany endpoint /last_entry
                    # Dla celów demonstracyjnych, pobieramy ostatni wjazd z logu zdarzeń.
                    try:
                        # To jest symulacja - w praktyce serwer powinien mieć endpoint GET /last_entry
                        # Zmieniamy to na odczyt lokalnej bazy danych serwera (wymaga uruchomienia z tego samego folderu)
                        import sqlite3
                        conn = sqlite3.connect('parking.db')
                        cursor = conn.cursor()
                        cursor.execute("SELECT license_plate FROM event_log WHERE event_type = 'ENTRY_GRANTED' ORDER BY timestamp DESC LIMIT 1")
                        last_plate_entry = cursor.fetchone()
                        conn.close()

                        if last_plate_entry:
                            last_plate = last_plate_entry[0]
                            # Sprawdzamy, czy ta tablica nie jest już przypisana
                            if last_plate not in plate_associations.values():
                                plate_associations[object_id] = last_plate
                                print(f"Powiązano ID: {object_id} z tablicą: {last_plate}")
                    except Exception as e:
                        print(f"Nie udało się pobrać ostatniego wjazdu: {e}")


        # Przygotowanie i wysłanie danych do serwera
        current_time = time.time()
        if current_time - last_update_time > UPDATE_INTERVAL_SECONDS:
            payload = {"vehicles": []}
            for object_id, box in tracked_objects.items():
                if object_id in plate_associations:
                    payload["vehicles"].append({
                        "license_plate": plate_associations[object_id],
                        "box": list(box)
                    })
            
            if payload["vehicles"]:
                try:
                    response = requests.post(f"{SERVER_URL}/update_positions", json=payload, timeout=2)
                    if response.status_code == 200:
                        print(f"[{time.strftime('%H:%M:%S')}] Wysłano pozycje {len(payload['vehicles'])} pojazdów do serwera.")
                    else:
                        print(f"Błąd serwera: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"Błąd połączenia z serwerem: {e}")

            last_update_time = current_time

        # Wizualizacja na klatce
        cv2.polylines(frame, [ENTRY_ZONE_POLY], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(frame, "Strefa wjazdu", (ENTRY_ZONE_POLY[0][0], ENTRY_ZONE_POLY[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        for object_id, box in tracked_objects.items():
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {object_id}"
            if object_id in plate_associations:
                label = f"{plate_associations[object_id]} (ID: {object_id})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Wyświetlenie klatki
        cv2.imshow("Kamera Gorna - Cyber Parking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()