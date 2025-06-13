import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr

image = cv2.imread("Przechwytywanie_tablica_rej.PNG")

def crop_margins(image: np.ndarray, margin: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    return image[margin:h-margin, margin:w-margin]

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
    kernel = np.array([[-1,-1,-1],
                       [-1,12,-1],
                       [-1,-1,-1]])
    # kernel = np.array([[0,-1,0], [-1,15,-1], [0,-1,0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    sharpened = crop_margins(sharpened, margin=20)
    
    # OPCJONALNIE: Binaryzacja (próg adaptacyjny jest lepszy od stałego)
    # Czasami pomaga, a czasami nie - warto przetestować.
    # binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)
    
    return sharpened # lub 'binary', jeśli zdecydujesz się na ten krok


# OCR z EasyOCR
def perform_easyocr(image: np.ndarray) -> str:
    reader = easyocr.Reader(['pl', 'en'])  # Obsługuje polskie znaki, jeśli występują
    result = reader.readtext(image)
    
    # Wyciągnij tylko teksty z wyników (ignorujemy współrzędne i prawdopodobieństwo na razie)
    texts = [entry[1] for entry in result]
    combined_text = ' '.join(texts)
    return combined_text.strip()



processed = preprocess_for_ocr(image)
recognized_text = perform_easyocr(processed)

print("Odczytany tekst (EasyOCR):", recognized_text)

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


