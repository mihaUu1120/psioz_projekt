# --- Plik Konfiguracyjny dla Systemu Monitorowania Parkingu ---

# --- Konfiguracja Modeli Detekcji ---
DETECTION_MODEL_PATH = "best_dziala_najlepiej.pt"
DETECTION_MODEL_PLATES_PATH = "best_plates.pt"

# --- Konfiguracja Źródeł Wideo (indeksy kamer) ---
VIDEO_SOURCE_BOTTOM = 2
VIDEO_SOURCE_TOP = 0
VIDEO_SOURCE_EXIT = 1

# --- Konfiguracja Klas Obiektów i Progów Pewności ---
TARGET_CLASS = "car"
PLATE_TARGET_CLASS = "plate"
CONFIDENCE_THRESHOLD = 0.50
PLATE_CONFIDENCE_THRESHOLD = 0.75

# --- Konfiguracja Miejsc Parkingowych ---
PARKING_ZONES = {
    "ZONE_1": (1340, 676, 1624, 810),
    "ZONE_2": (1335, 531, 1610, 664),
    "ZONE_3": (1325, 386, 1595, 519),
    "ZONE_4": (1077, 83, 1215, 321),
    "ZONE_5": (934, 78, 1067, 323),
    "ZONE_6": (797, 76, 922, 321),
    "ZONE_7": (658, 77, 774, 316),
    "ZONE_8": (258, 379, 527, 505),
    "ZONE_9": (235, 524, 509, 650),
    "ZONE_10": (224, 671, 499, 803)
}
TOTAL_PARKING_SPOTS = len(PARKING_ZONES)
PARKING_OVERLAP_THRESHOLD = 0.2 # Minimalne pokrycie, aby uznać samochód w miejscu parkingowym

# --- Konfiguracja Stref Drogowych ---
ROAD_ZONES = {
    "ROAD_1": (995, 849, 1231, 1029),
    "ROAD_2": (994, 359, 1289, 845),
    "ROAD_3": (550, 358, 994, 537),
    "ROAD_4": (550, 538, 833, 832),
    "ROAD_5": (602, 835, 831, 1028)
}

ROAD_OVERLAP_THRESHOLD = 0.90 # Minimalne pokrycie, aby uznać samochód na drodze
PARKED_OVERLAP_THRESHOLD = 0.50 # Minimalne pokrycie, aby uznać samochód za zaparkowany

# --- Ustawienia Rozdzielczości Kamery ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- Konfiguracja Strefy Wjazdowej ---
OVERLAP_THRESHOLD = 0.80 # Ogólny próg pokrycia dla stref
ENTRYPOINT_ZONE = (1280, 852, 1642, 1016)
ENTRY_GATE_LIGHT = (1230, 845, 1248, 1023) # Wskaźnik wizualny dla światła bramy wjazdowej

# --- Konfiguracja Strefy Wyjazdowej (dla kamery górnej) ---
EXITPOINT_ZONE = (198, 842, 534, 989)
EXIT_GATE_LIGHT = (188, 825, 199, 993) # Wskaźnik wizualny dla światła bramy wyjazdowej

# --- Próg Pokrycia dla Kolizji ---
COLLISION_OVERLAP_THRESHOLD = 0.10

# --- Konfiguracja Ładowania Stanu i Przypisywania ---
INITIALIZATION_FRAMES = 100 # Liczba początkowych klatek, aby tracker się ustabilizował
REASSIGNMENT_DISTANCE_THRESHOLD = 200 # Maksymalna odległość dla ponownego przypisania znanych tablic do nowych śladów

# --- Konfiguracja Odświeżania Bazy Danych ---
DB_RELOAD_INTERVAL_FRAMES = 30 # Jak często ładować dane pojazdów z bazy danych

# --- Konfiguracja Timera Bramy Wjazdowej ---
GATE_TIMEOUT = 10 # Sekundy, po których brama zamyka się po spełnieniu warunków otwarcia
