import cv2
import os

# --- Konfiguracja ---
ID_KAMERY = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CROP_SIZE = 1080
SAVED_SIZE = 640
ZDJECIA_FOLDER = '.\images'

# --- Zmienne globalne ---
dragging = False
crop_x, crop_y = 100, 100  # Początkowa pozycja crop boxa
frame = None

def znajdz_nastepny_numer(folder, prefix="image_", extension=".jpg"):
    numer = 1
    while os.path.exists(os.path.join(folder, f"{prefix}{numer}{extension}")):
        numer += 1
    return numer

def mouse_callback(event, x, y, flags, param):
    global dragging, crop_x, crop_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if crop_x <= x <= crop_x + CROP_SIZE and crop_y <= y <= crop_y + CROP_SIZE:
            dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        crop_x = max(0, min(x - CROP_SIZE // 2, FRAME_WIDTH - CROP_SIZE))
        crop_y = max(0, min(y - CROP_SIZE // 2, FRAME_HEIGHT - CROP_SIZE))

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# --- Inicjalizacja kamery ---
cap = cv2.VideoCapture(ID_KAMERY)
if not cap.isOpened():
    print(f"Błąd: Nie można otworzyć kamery o ID: {ID_KAMERY}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cv2.namedWindow("Podglad")
cv2.setMouseCallback("Podglad", mouse_callback)

print("Naciśnij 's', aby zapisać wycięty fragment 640x640.")
print("Naciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie można odczytać klatki z kamery.")
        break

    # Rysowanie prostokąta crop boxa
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (crop_x, crop_y), (crop_x + CROP_SIZE, crop_y + CROP_SIZE), (0, 255, 0), 2)

    cv2.imshow("Podglad", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        # Wycięcie fragmentu obrazu
        crop = frame[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]
        crop_resized = cv2.resize(crop, (SAVED_SIZE, SAVED_SIZE))
        numer = znajdz_nastepny_numer(ZDJECIA_FOLDER)
        filename = os.path.join(ZDJECIA_FOLDER, f"image_{numer}.jpg")
        cv2.imwrite(filename, crop_resized)
        print(f"Zapisano: {filename}")

cap.release()
cv2.destroyAllWindows()
