import cv2
import easyocr
from ultralytics import YOLO
import sqlite3

# Połączenie z bazą i funkcja sprawdzająca
conn = sqlite3.connect('plates.db')
cursor = conn.cursor()

def is_plate_in_db(plate):
    cursor.execute("SELECT 1 FROM plates WHERE plate_number = ?", (plate,))
    return cursor.fetchone() is not None

model = YOLO("yolov8n.pt")
ocr = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % 10 == 0:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pojazd_crop = frame[y1:y2, x1:x2]

                wyniki = ocr.readtext(pojazd_crop)

                plate_found = False
                plate_text = ""

                for (bbox, text, conf) in wyniki:
                    czysty_tekst = text.replace(" ", "").upper()
                    if 4 <= len(czysty_tekst) <= 10:
                        plate_found = True
                        plate_text = czysty_tekst
                        print(f"Odczytano: {czysty_tekst} (pewność: {conf:.2f})")

                        (tl, tr, br, bl) = bbox
                        tl = tuple(map(int, tl))
                        br = tuple(map(int, br))
                        cv2.rectangle(pojazd_crop, tl, br, (0, 255, 0), 2)
                        cv2.putText(pojazd_crop, czysty_tekst, (tl[0], tl[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Zakładam, że interesuje Cię tylko pierwsza pasująca tablica
                        break

                # Kolor ramki zależy od tego, czy tablica jest w bazie
                if plate_found:
                    if is_plate_in_db(plate_text):
                        color = (0, 255, 0)  # zielony
                    else:
                        color = (0, 0, 255)  # czerwony
                else:
                    color = (255, 0, 0)  # niebieski (lub inny, jeśli nie wykryto tablicy)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.imshow("Kamera - wykrywanie tablic", frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
