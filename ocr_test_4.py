import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ścieżka do obrazu
image_path = '.\zapisane_tablice\80902_2.jpg'  # np. 'tablica.jpg'

# Inicjalizacja OCR
reader = easyocr.Reader(['en'])

# Wczytanie obrazu do OpenCV (BGR)
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# OCR - pierwszy przebieg
results = reader.readtext(image_rgb)

# Rysowanie bboxów na kopii obrazu
image_with_boxes = image_rgb.copy()
for bbox, text, prob in results:
    pts = np.array(bbox).astype(int)
    cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(image_with_boxes, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Wyświetlenie obrazu z bboxami (pierwszy przelot)
plt.figure(figsize=(10, 8))
plt.title("Pierwszy przelot OCR z bbox")
plt.imshow(image_with_boxes)
plt.axis('off')
plt.show()

# OCR na każdym wyciętym fragmencie
for i, (bbox, text, prob) in enumerate(results):
    pts = np.array(bbox).astype(int)
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    # Wycinanie bboxa
    cropped = image_rgb[y_min:y_max, x_min:x_max]

    # OCR na wyciętym fragmencie
    second_result = reader.readtext(cropped)

    # Rysowanie bboxów na fragmencie
    cropped_with_boxes = cropped.copy()
    for bbox2, text2, prob2 in second_result:
        pts2 = np.array(bbox2).astype(int)
        cv2.polylines(cropped_with_boxes, [pts2], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(cropped_with_boxes, text2, tuple(pts2[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Wyświetlenie z bboxami
    plt.figure(figsize=(4, 3))
    plt.imshow(cropped_with_boxes)
    plt.axis('off')
    title = f"Wycięty fragment {i+1}"
    if second_result:
        title += f" - OCR: {second_result[0][1]} (Pewność: {second_result[0][2]:.2f})"
    else:
        title += " - OCR: brak wyniku"
    plt.title(title)
    plt.show()
