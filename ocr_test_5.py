import cv2
import easyocr
import numpy as np
import os

# === Inicjalizacja ===,
reader = easyocr.Reader(['en'])
folder_path = 'zapisane_tablice'

# === Przetwarzanie wszystkich plików w folderze ===,
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(folder_path, filename)
        print(f"\n Przetwarzam: {filename}")

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours,_  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 20 < h < 100 and 10 < w < 80 and w / h < 1.0:
                char_regions.append((x, y, w, h))

        char_regions = sorted(char_regions, key=lambda b: b[0])

        recognized = ''
        for (x, y, w, h) in char_regions:
            char_img = gray[y:y + h, x:x + w]
            char_img = cv2.copyMakeBorder(char_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

            result = reader.readtext(char_img, detail=0, paragraph=False)
            if result:
                recognized += result[0]
            else:
                recognized += '?'

        print(" Rozpoznany tekst:", recognized)