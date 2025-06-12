import cv2
import sys

# --- Konfiguracja ---
# Wpisz ID kamery, na której chcesz wybrać strefę.
# Dla górnej kamery w Twoim głównym skrypcie to było ID = 1.
ID_KAMERY = 1 
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Globalne zmienne do przechowywania punktów i stanu rysowania
ref_point = []
drawing = False

def select_zone_callback(event, x, y, flags, param):
    """Funkcja zwrotna obsługująca zdarzenia myszy."""
    global ref_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # Rozpocznij rysowanie po wciśnięciu lewego przycisku myszy
        ref_point = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        # Zakończ rysowanie po puszczeniu przycisku
        ref_point.append((x, y))
        drawing = False

        # Narysuj finalny prostokąt na obrazie
        cv2.rectangle(frame, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Wybierz strefe", frame)


# --- Główna część programu ---
cap = cv2.VideoCapture(ID_KAMERY)

if not cap.isOpened():
    print(f"Błąd: Nie można otworzyć kamery o ID: {ID_KAMERY}")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Uruchomiono podgląd z kamery.")
print("Naciśnij klawisz 's', aby zapisać klatkę i przejść do wyboru strefy.")
print("Naciśnij klawisz 'q', aby zamknąć.")

frame = None

# Pętla do przechwycenia idealnej klatki
while True:
    ret, current_frame = cap.read()
    if not ret:
        print("Błąd: Nie można odczytać klatki z kamery.")
        break
    
    cv2.imshow("Podglad na zywo - nacisnij 's' aby zapisac klatke", current_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
    elif key == ord('s'):
        frame = current_frame
        print("Klatka zapisana. Możesz teraz wybrać strefę.")
        break

cv2.destroyAllWindows()
cap.release()

if frame is not None:
    clone = frame.copy()
    cv2.namedWindow("Wybierz strefe")
    cv2.setMouseCallback("Wybierz strefe", select_zone_callback)

    print("\n--- INSTRUKCJA ---")
    print("1. Kliknij i przytrzymaj lewy przycisk myszy, aby rozpocząć rysowanie prostokąta.")
    print("2. Przeciągnij mysz, aby narysować strefę.")
    print("3. Puść przycisk myszy, aby zakończyć.")
    print("4. Współrzędne pojawią się w konsoli.")
    print("5. Naciśnij 'r', aby zresetować wybór.")
    print("6. Naciśnij 'q', aby zakończyć i wyświetlić finalne współrzędne.")

    while True:
        cv2.imshow("Wybierz strefe", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            # Resetuj wybór po naciśnięciu 'r'
            frame = clone.copy()
            ref_point = []
            print("Wybór zresetowany. Możesz rysować od nowa.")
        
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Wyświetl finalne, uporządkowane współrzędne
    if len(ref_point) == 2:
        x1 = min(ref_point[0][0], ref_point[1][0])
        y1 = min(ref_point[0][1], ref_point[1][1])
        x2 = max(ref_point[0][0], ref_point[1][0])
        y2 = max(ref_point[0][1], ref_point[1][1])
        
        final_coords = (x1, y1, x2, y2)
        
        print("\n=======================================================")
        print("Twoje współrzędne strefy są gotowe do skopiowania!")
        print(f"ENTRYPOINT_ZONE = {final_coords}")
        print("=======================================================")
    else:
        print("\nNie wybrano strefy.")