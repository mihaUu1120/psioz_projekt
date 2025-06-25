import cv2
import numpy as np
import importlib.util
import os

# --- Ladowanie konfiguracji z config.py ---
def load_config_from_file(config_path="ladne/config.py"):
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None:
        print(f"Blad: Nie znaleziono pliku konfiguracyjnego (sciezka: {config_path})")
        print("Upewnij sie, ze plik config.py istnieje w tym samym katalogu.")
        exit()
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

print("Laduje konfiguracje z config.py...")
try:
    current_config = load_config_from_file()
except Exception as e:
    print(f"Blad podczas ladowania config.py: {e}")
    print("Upewnij sie, ze config.py jest poprawnym plikiem Pythona.")
    exit()

# Upewnij sie, ze wymagane zmienne sa dostepne
try:
    VIDEO_SOURCE_TOP = getattr(current_config, 'VIDEO_SOURCE_TOP', 0)
    FRAME_WIDTH = getattr(current_config, 'FRAME_WIDTH', 1280)
    FRAME_HEIGHT = getattr(current_config, 'FRAME_HEIGHT', 720)
    # Wczytaj istniejace strefy lub zainicjuj puste, jesli ich nie ma
    INITIAL_PARKING_ZONES = getattr(current_config, 'PARKING_ZONES', {})
    INITIAL_ROAD_ZONES = getattr(current_config, 'ROAD_ZONES', {})
    INITIAL_ENTRYPOINT_ZONE = getattr(current_config, 'ENTRYPOINT_ZONE', None)
    INITIAL_EXITPOINT_ZONE = getattr(current_config, 'EXITPOINT_ZONE', None)
    INITIAL_ENTRY_GATE_LIGHT = getattr(current_config, 'ENTRY_GATE_LIGHT', None)
    INITIAL_EXIT_GATE_LIGHT = getattr(current_config, 'EXIT_GATE_LIGHT', None)

    # Przygotuj slownik do przechowywania edytowanych stref
    CONFIG_ZONES = {
        "PARKING_ZONES": dict(INITIAL_PARKING_ZONES), # Kopiuj, zeby edytowac
        "ROAD_ZONES": dict(INITIAL_ROAD_ZONES),
        "ENTRYPOINT_ZONE": INITIAL_ENTRYPOINT_ZONE,
        "EXITPOINT_ZONE": INITIAL_EXITPOINT_ZONE,
        "ENTRY_GATE_LIGHT": INITIAL_ENTRY_GATE_LIGHT,
        "EXIT_GATE_LIGHT": INITIAL_EXIT_GATE_LIGHT,
    }

except AttributeError as e:
    print(f"Blad: Wymagana zmienna konfiguracyjna nie znaleziona w config.py: {e}")
    print("Sprawdz, czy config.py zawiera definicje VIDEO_SOURCE_TOP, FRAME_WIDTH, FRAME_HEIGHT oraz stref.")
    exit()

# --- Definicja typow stref i ich kolejnosci ---
# (klucz do CONFIG_ZONES, wyswietlana nazwa, maksymalna liczba instancji (-1 dla wielu))
ZONE_TYPES_ORDER = [
    ("PARKING_ZONES", "Parking Zone", -1),
    ("ROAD_ZONES", "Road Zone", -1),
    ("ENTRYPOINT_ZONE", "Entry Point Zone", 1),
    ("EXITPOINT_ZONE", "Exit Point Zone", 1),
    ("ENTRY_GATE_LIGHT", "Entry Gate Light", 1),
    ("EXIT_GATE_LIGHT", "Exit Gate Light", 1),
]

current_zone_type_index = 0
current_zone_name_display = ""
current_zone_config_key = ""
max_instances = 0
drawing = False
ix, iy = -1, -1
temp_frame = None # Uzywane do rysowania prostokata podczas przeciagania mysza
original_frame_copy = None # Przechowuje oryginalna klatke bez tymczasowych rysunkow
current_zone_counter = 0 # Licznik dla nazw stref wielokrotnych

# --- Funkcja pomocnicza do okreslania koloru strefy ---
def get_zone_color(key):
    if "PARKING" in key:
        return (0, 255, 255)  # Zolty dla parkowania
    elif "ROAD" in key:
        return (255, 0, 255)  # Magenta dla drogi
    elif "ENTRYPOINT" in key:
        return (255, 0, 0)    # Niebieski dla wjazdu
    elif "EXITPOINT" in key:
        return (0, 0, 255)    # Czerwony dla wyjazdu
    elif "GATE_LIGHT" in key:
        return (0, 255, 0)    # Zielony dla swiatel bramy
    return (255, 255, 255) # Bialy jako domyslny

# --- Callback funkcji myszy ---
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, temp_frame, current_zone_counter, original_frame_copy

    # Gdy kliknieto lewy przycisk myszy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Gdy przycisk myszy zostal puszczony
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)

        if x1 == x2 or y1 == y2: # Zapobiegaj rysowaniu pustych prostokatow
            print("Too small rectangle. Try again.")
            return

        if current_zone_config_key in ["PARKING_ZONES", "ROAD_ZONES"]:
            # Dla stref wielokrotnych, generuj unikalna nazwe
            zone_name = f"{current_zone_name_display.replace(' ', '_').upper().replace('ZONE_', '')}_{current_zone_counter}"
            CONFIG_ZONES[current_zone_config_key][zone_name] = (x1, y1, x2, y2)
            print(f"Defined {current_zone_config_key}: '{zone_name}' = ({x1}, {y1}, {x2}, {y2})")
            current_zone_counter += 1
        else:
            # Dla stref pojedynczych, przypisz bezposrednio
            CONFIG_ZONES[current_zone_config_key] = (x1, y1, x2, y2)
            print(f"Defined {current_zone_config_key}: '{current_zone_name_display}' = ({x1}, {y1}, {x2}, {y2})")

        # Jesli to strefa pojedyncza, przejdz do nastepnego typu strefy
        if max_instances == 1:
            next_zone_type()
        
        # temp_frame zostanie odswiezony w glownej petli, wiec nie trzeba tutaj

    # Gdy mysz jest przeciagana i rysowanie aktywne
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_frame = original_frame_copy.copy() # Zawsze zaczynaj od czystej kopii
        cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2) # Rysuj zielony prostokat
        cv2.putText(temp_frame, "Drawing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# --- Funkcja do przelaczania typu strefy ---
def next_zone_type():
    global current_zone_type_index, current_zone_name_display, current_zone_config_key, max_instances, current_zone_counter, temp_frame, original_frame_copy

    current_zone_type_index = (current_zone_type_index + 1) % len(ZONE_TYPES_ORDER)
    current_zone_config_key, current_zone_name_display, max_instances = ZONE_TYPES_ORDER[current_zone_type_index]
    
    # Dla stref wielokrotnych, zainicjuj licznik na podstawie juz istniejacych
    if current_zone_config_key in ["PARKING_ZONES", "ROAD_ZONES"]:
        current_zone_counter = len(CONFIG_ZONES[current_zone_config_key])
    else:
        current_zone_counter = 0 # Nieistotne dla stref pojedynczych

    print(f"\n--- Current Zone Type: {current_zone_name_display} ---")
    if max_instances == -1:
        print("Click and drag to define a zone. You can define multiple zones of this type.")
        print("Press 'N' for next type, 'C' to clear current type, 'S' to save, 'Q' to quit.")
    else:
        print("Click and drag to define this zone. It will automatically move to next type after definition.")
        print("Press 'C' to clear current type, 'S' to save, 'Q' to quit.")

    # Odswiez temp_frame po zmianie typu strefy, aby pokazac tylko aktualne strefy
    if original_frame_copy is not None:
        temp_frame = original_frame_copy.copy()

# --- Funkcja do czyszczenia aktualnie edytowanego typu strefy ---
def clear_current_zone_type():
    global CONFIG_ZONES, current_zone_counter
    if current_zone_config_key in ["PARKING_ZONES", "ROAD_ZONES"]:
        CONFIG_ZONES[current_zone_config_key].clear()
        print(f"Cleared all defined '{current_zone_name_display}' zones.")
        current_zone_counter = 0 # Zresetuj licznik po wyczyszczeniu
    else:
        CONFIG_ZONES[current_zone_config_key] = None
        print(f"Cleared defined '{current_zone_name_display}' zone.")
    
    global temp_frame, original_frame_copy
    if original_frame_copy is not None:
        temp_frame = original_frame_copy.copy() # Odswiez obraz po wyczyszczeniu

# --- Zapis konfiguracji do pliku ---
def save_config(filename="new_config_zones.py"):
    with open(filename, "w", encoding='utf-8') as f:
        f.write("# This file was generated by the zone configuration script.\n")
        f.write("# It contains the defined rectangular zones for your parking system.\n\n")

        # Zapisz zmienne kamery
        f.write(f"VIDEO_SOURCE_TOP = {VIDEO_SOURCE_TOP}\n")
        f.write(f"FRAME_WIDTH = {FRAME_WIDTH}\n")
        f.write(f"FRAME_HEIGHT = {FRAME_HEIGHT}\n\n")


        for key, value in CONFIG_ZONES.items():
            if isinstance(value, dict):
                f.write(f"{key} = {{\n")
                # Sortuj klucze dla lepszej czytelnosci i powtarzalnosci
                for sub_key in sorted(value.keys()):
                    sub_value = value[sub_key]
                    f.write(f"    '{sub_key}': {sub_value},\n")
                f.write("}\n\n")
            else:
                f.write(f"{key} = {value}\n\n")
    print(f"\nConfiguration saved to file: {filename}")
    print("You can now copy the contents of this file to your config.py.")

# --- Glowna logika programu ---
cap = cv2.VideoCapture(VIDEO_SOURCE_TOP)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"Blad: Nie mozna otworzyc zrodla wideo: {VIDEO_SOURCE_TOP}. Sprawdz konfiguracje kamery w config.py.")
    exit()

cv2.namedWindow("Zone Configurator")
cv2.setMouseCallback("Zone Configurator", draw_rectangle)

print("Welcome to the Zone Configurator!")
next_zone_type() # Zainicjuj pierwszy typ strefy

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check camera connection or video source.")
            break

        original_frame_copy = frame.copy() # Zawsze aktualizuj kopie oryginalnej klatki

        # Jesli nie rysujemy, temp_frame to aktualna klatka
        if not drawing:
            temp_frame = original_frame_copy.copy()

        display_frame = temp_frame.copy()

        # Rysuj aktualnie edytowane strefy
        current_zone_color = get_zone_color(current_zone_config_key)
        if current_zone_config_key in ["PARKING_ZONES", "ROAD_ZONES"]: # Strefy wielokrotne
            for zone_name, coords in CONFIG_ZONES[current_zone_config_key].items():
                x1, y1, x2, y2 = coords
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), current_zone_color, 2)
                cv2.putText(display_frame, zone_name.replace('_', ' '), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_zone_color, 1)
        elif CONFIG_ZONES[current_zone_config_key] is not None: # Strefy pojedyncze
            x1, y1, x2, y2 = CONFIG_ZONES[current_zone_config_key]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), current_zone_color, 2)
            cv2.putText(display_frame, current_zone_name_display, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_zone_color, 1)

        # DODATKOWO: Rysuj Entry/Exit Gate Light (zawsze widoczne)
        gate_light_color = get_zone_color("GATE_LIGHT") # Uzyj koloru dla Gate Light
        if CONFIG_ZONES["ENTRY_GATE_LIGHT"] is not None:
            x1, y1, x2, y2 = CONFIG_ZONES["ENTRY_GATE_LIGHT"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), gate_light_color, 2)
            cv2.putText(display_frame, "Entry Gate Light", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gate_light_color, 1)
        
        if CONFIG_ZONES["EXIT_GATE_LIGHT"] is not None:
            x1, y1, x2, y2 = CONFIG_ZONES["EXIT_GATE_LIGHT"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), gate_light_color, 2)
            cv2.putText(display_frame, "Exit Gate Light", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gate_light_color, 1)


        # Wyswietl aktualne instrukcje
        cv2.putText(display_frame, f"Defining: {current_zone_name_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_frame, "N: Next zone type | C: Clear current type", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(display_frame, "S: Save to file | Q: Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


        cv2.imshow("Zone Configurator", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            next_zone_type()
        elif key == ord('c'): # Nowa opcja: wyczysc aktualny typ strefy
            clear_current_zone_type()
        elif key == ord('s'):
            save_config()
        elif key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Zone configuration finished.")