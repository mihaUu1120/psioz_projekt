import cv2
import sys

# --- Konfiguracja ---
ID_KAMERY = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
display_scale = 1  # np. 0.5 dla zmniejszenia rozmiaru wyświetlania

# Globalne zmienne
zones = []
drawing = False
start_point = None
frame = None
clone = None

def scale_point(x, y, scale):
    return int(x / scale), int(y / scale)

def select_zone_callback(event, x, y, flags, param):
    """Funkcja obsługująca zdarzenia myszy."""
    global drawing, start_point, frame, zones, clone

    # Przeskaluj współrzędne do oryginalnych
    x_scaled, y_scaled = scale_point(x, y, display_scale)
    temp_frame = clone.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x_scaled, y_scaled)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(temp_frame, start_point, (x_scaled, y_scaled), (0, 255, 0), 2)
        for rect in zones:
            cv2.rectangle(temp_frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        temp_display = cv2.resize(temp_frame, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Wybierz strefy", temp_display)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x_scaled, y_scaled)

        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        zones.append((x1, y1, x2, y2))

        frame = clone.copy()
        for rect in zones:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        clone = frame.copy()
        temp_display = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Wybierz strefy", temp_display)

# --- Główna część programu ---
cap = cv2.VideoCapture(ID_KAMERY)

if not cap.isOpened():
    print(f"Błąd: Nie można otworzyć kamery o ID: {ID_KAMERY}")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Uruchomiono podgląd z kamery.")
print("Naciśnij klawisz 's', aby zapisać klatkę i przejść do wyboru stref.")
print("Naciśnij klawisz 'q', aby zamknąć.")

while True:
    ret, current_frame = cap.read()
    if not ret:
        print("Błąd: Nie można odczytać klatki z kamery.")
        break

    display_frame = cv2.resize(current_frame, None, fx=display_scale, fy=display_scale)
    cv2.imshow("Podglad na zywo - nacisnij 's' aby zapisac klatke", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
    elif key == ord('s'):
        frame = current_frame.copy()
        break

cv2.destroyAllWindows()
cap.release()

if frame is not None:
    clone = frame.copy()
    cv2.namedWindow("Wybierz strefy")
    cv2.setMouseCallback("Wybierz strefy", select_zone_callback)

    print("\n--- INSTRUKCJA ---")
    print("1. Kliknij i przeciągnij, aby zaznaczyć strefę.")
    print("2. Możesz dodać wiele stref.")
    print("3. Naciśnij 'r', aby zresetować wszystkie strefy.")
    print("4. Naciśnij 'q', aby zakończyć i wypisać współrzędne.")

    while True:
        display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Wybierz strefy", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            zones = []
            frame = clone.copy()
            print("Zresetowano wszystkie strefy.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if zones:
        print("\n=======================================================")
        print("Twoje współrzędne stref są gotowe do skopiowania!")
        for i, rect in enumerate(zones, 1):
            print(f"ZONE_{i} = {rect}")
        print("=======================================================")
    else:
        print("\nNie wybrano żadnych stref.")
