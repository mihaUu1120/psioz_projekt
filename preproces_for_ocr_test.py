import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread("Przechwytywanie_tablica_rej.PNG")

def crop_margins(image: np.ndarray, margin: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    return image[margin:h-margin, margin:w-margin]

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Przygotowuje obraz wyciętej tablicy rejestracyjnej do OCR.
    """
    h, w = image.shape[:2]
    if h < 50:
        scale_factor = 100 / h
        width = int(w * scale_factor)
        height = int(h * scale_factor)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    blurred = cv2.medianBlur(contrast_enhanced, 3)

    kernel = np.array([[-1,-1,-1],
                       [-1,12,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    sharpened = crop_margins(sharpened, margin=20)
    
    return sharpened

# OCR z pytesseract
def perform_tesseract(image: np.ndarray) -> str:
    # pytesseract oczekuje obrazu w skali szarości lub RGB, jeśli jest BGR to konwertujemy:
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    # Konfiguracja dla Tesseract (opcjonalnie, można dopasować do tablic rejestracyjnych)
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    text = pytesseract.image_to_string(img_rgb, lang='eng+pol', config=config)
    return text.strip()

processed = preprocess_for_ocr(image)
recognized_text = perform_tesseract(processed)

print("Odczytany tekst (Tesseract):", recognized_text)

# Pokaż efekt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Po przetwarzaniu")
plt.imshow(processed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
