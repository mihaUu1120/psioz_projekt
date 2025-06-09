import cv2
import easyocr
from ultralytics import YOLO

# Inicjalizacje
model = YOLO("yolov8n.pt")
ocr = easyocr.Reader(['en'])

# Otwórz kamerę (0 = kamera domyślna)
cap = cv2.VideoCapture(0)  # Możesz tu podać też ścieżkę do pliku

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wykrywanie obiektów YOLO
    results = model(img_rgb, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pojazd_crop = frame[y1:y2, x1:x2]

            # OCR tylko na fragmencie pojazdu
            wyniki = ocr.readtext(pojazd_crop)

            for (bbox, text, conf) in wyniki:
                czysty_tekst = text.replace(" ", "").upper()
                if 4 <= len(czysty_tekst) <= 10:
                    print(f"Odczytano: {czysty_tekst} (pewność: {conf:.2f})")

                    # Rysuj OCR na pojeździe
                    (tl, tr, br, bl) = bbox
                    tl = tuple(map(int, tl))
                    br = tuple(map(int, br))
                    cv2.rectangle(pojazd_crop, tl, br, (0, 255, 0), 2)
                    cv2.putText(pojazd_crop, czysty_tekst, (tl[0], tl[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Rysuj ramkę pojazdu na oryginale
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Wyświetl podgląd
    cv2.imshow("Kamera - wykrywanie tablic", frame)

    # Wyjście: Q lub ESC
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()