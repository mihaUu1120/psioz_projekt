import flask
from flask import request, jsonify
import sqlite3
import datetime
import json
import os

# --- Konfiguracja ---
DATABASE_FILE = 'parking.db'
# W osobnym pliku JSON lub tutaj dla prostoty
# Współrzędne wielokątów definiujących strefy na parkingu
ZONES_CONFIG = {
    "spots": [
        {"id": "P1", "poly": [[1339, 674], [1615, 674], [1615, 826], [1339, 826]]},
        {"id": "P2", "poly": [[1331, 520], [1604, 520], [1604, 656], [1331, 656]]},
    ],
    # "roads": [
    #     {"id": "R1", "poly": [[90, 210], [320, 210], [320, 280], [90, 280]]}
    # ]
}
PARKING_CAPACITY = 11 # Maksymalna pojemność parkingu

# Inicjalizacja aplikacji Flask
app = flask.Flask(__name__)

# --- Funkcje pomocnicze do geometrii ---

def get_car_center(box):
    """Oblicza środek ramki otaczającej [x1, y1, x2, y2]."""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center, y_center)

def is_point_in_polygon(point, polygon):
    """Sprawdza, czy punkt (x, y) znajduje się wewnątrz wielokąta."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# --- Funkcje bazy danych ---

def db_connect():
    """Nawiązuje połączenie z bazą danych."""
    return sqlite3.connect(DATABASE_FILE)

def init_db():
    """Tworzy tabele w bazie danych, jeśli nie istnieją i dodaje dane testowe."""
    if os.path.exists(DATABASE_FILE):
        print("Baza danych już istnieje.")
        return

    print("Tworzenie nowej bazy danych...")
    conn = db_connect()
    cursor = conn.cursor()

    # Tabela uprawnionych pojazdów
    cursor.execute('''
    CREATE TABLE authorized_vehicles (
        license_plate TEXT PRIMARY KEY NOT NULL
    )
    ''')

    # Tabela przechowująca aktualny stan parkingu (kto jest w środku)
    # Zapewnia odporność na restart serwera
    cursor.execute('''
    CREATE TABLE parking_state (
        license_plate TEXT PRIMARY KEY NOT NULL,
        entry_time TEXT NOT NULL
    )
    ''')
    
    # Dziennik wszystkich zdarzeń
    cursor.execute('''
    CREATE TABLE event_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        event_type TEXT NOT NULL, -- 'ENTRY_REQUEST', 'ENTRY_GRANTED', 'ENTRY_DENIED', 'EXIT_REQUEST', 'EXIT_GRANTED'
        license_plate TEXT NOT NULL,
        details TEXT
    )
    ''')

    # Dziennik naruszeń
    cursor.execute('''
    CREATE TABLE violations_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        license_plate TEXT NOT NULL,
        violation_type TEXT NOT NULL, -- 'ILLEGAL_PARKING', 'BLOCKING_ROAD'
        position TEXT
    )
    ''')

    # Dodanie przykładowych danych
    sample_plates = [('KR12345',), ('WE54321',), ('PO67890',)]
    cursor.executemany('INSERT INTO authorized_vehicles VALUES (?)', sample_plates)
    
    conn.commit()
    conn.close()
    print("Baza danych została zainicjowana z przykładowymi danymi.")


# --- Endpointy API ---

@app.route('/entry_request', methods=['POST'])
def handle_entry_request():
    """Obsługuje prośbę o wjazd od kamery wjazdowej."""
    data = request.get_json()
    plate = data.get('license_plate', '').upper()
    timestamp = datetime.datetime.now().isoformat()

    conn = db_connect()
    cursor = conn.cursor()

    # Logowanie próby wjazdu
    cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                   (timestamp, 'ENTRY_REQUEST', plate, ''))
    conn.commit()

    # 1. Sprawdzenie autoryzacji
    cursor.execute("SELECT 1 FROM authorized_vehicles WHERE license_plate = ?", (plate,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                       (timestamp, 'ENTRY_DENIED', plate, 'Brak autoryzacji'))
        conn.commit()
        conn.close()
        return jsonify({"decision": "DENY", "reason": "Pojazd nieuprawniony"}), 403

    # 2. Sprawdzenie, czy pojazd już jest na parkingu
    cursor.execute("SELECT 1 FROM parking_state WHERE license_plate = ?", (plate,))
    if cursor.fetchone():
        cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                       (timestamp, 'ENTRY_DENIED', plate, 'Pojazd już jest na parkingu'))
        conn.commit()
        conn.close()
        return jsonify({"decision": "DENY", "reason": "Pojazd już jest na parkingu"}), 409

    # 3. Sprawdzenie, czy jest wolne miejsce
    cursor.execute("SELECT COUNT(*) FROM parking_state")
    current_occupancy = cursor.fetchone()[0]
    if current_occupancy >= PARKING_CAPACITY:
        cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                       (timestamp, 'ENTRY_DENIED', plate, 'Parking pełny'))
        conn.commit()
        conn.close()
        return jsonify({"decision": "DENY", "reason": "Parking pełny"}), 409
    
    # Zgoda na wjazd - aktualizacja stanu i logowanie
    cursor.execute("INSERT INTO parking_state (license_plate, entry_time) VALUES (?, ?)", (plate, timestamp))
    cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                   (timestamp, 'ENTRY_GRANTED', plate, 'Otwarto bramkę wjazdową'))
    conn.commit()
    conn.close()
    
    return jsonify({"decision": "OPEN"}), 200

@app.route('/exit_request', methods=['POST'])
def handle_exit_request():
    """Obsługuje prośbę o wyjazd od kamery wyjazdowej."""
    data = request.get_json()
    plate = data.get('license_plate', '').upper()
    timestamp = datetime.datetime.now().isoformat()

    conn = db_connect()
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                   (timestamp, 'EXIT_REQUEST', plate, ''))
    conn.commit()

    # Sprawdzenie, czy pojazd jest na parkingu
    cursor.execute("SELECT 1 FROM parking_state WHERE license_plate = ?", (plate,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                       (timestamp, 'EXIT_DENIED', plate, 'Pojazdu nie ma na parkingu'))
        conn.commit()
        conn.close()
        return jsonify({"decision": "DENY", "reason": "Tego pojazdu nie ma na parkingu"}), 404
        
    # Zgoda na wyjazd - usunięcie ze stanu i logowanie
    cursor.execute("DELETE FROM parking_state WHERE license_plate = ?", (plate,))
    cursor.execute("INSERT INTO event_log (timestamp, event_type, license_plate, details) VALUES (?, ?, ?, ?)",
                   (timestamp, 'EXIT_GRANTED', plate, 'Otwarto bramkę wyjazdową'))
    conn.commit()
    conn.close()

    return jsonify({"decision": "OPEN"}), 200

@app.route('/update_positions', methods=['POST'])
def update_positions():
    """Odbiera i analizuje pozycje aut z kamery górnej."""
    vehicles_data = request.get_json().get('vehicles', [])
    timestamp = datetime.datetime.now().isoformat()
    
    conn = db_connect()
    cursor = conn.cursor()

    for vehicle in vehicles_data:
        plate = vehicle.get('license_plate')
        box = vehicle.get('box')
        if not plate or not box:
            continue
            
        center_point = get_car_center(box)

        # Sprawdzenie, czy auto blokuje drogę
        for road_zone in ZONES_CONFIG['roads']:
            if is_point_in_polygon(center_point, road_zone['poly']):
                # UWAGA: W systemie produkcyjnym tutaj należałoby sprawdzić,
                # czy auto jest w tej strefie przez określony czas (np. > 30s)
                # zanim zaloguje się naruszenie.
                print(f"ALARM: Pojazd {plate} blokuje drogę {road_zone['id']}!")
                cursor.execute(
                    "INSERT INTO violations_log (timestamp, license_plate, violation_type, position) VALUES (?, ?, ?, ?)",
                    (timestamp, plate, 'BLOCKING_ROAD', json.dumps(box))
                )
                conn.commit()
    
    conn.close()
    return jsonify({"status": "positions_updated", "processed_vehicles": len(vehicles_data)}), 200

# --- Uruchomienie serwera ---
if __name__ == '__main__':
    init_db()
    # Użyj host='0.0.0.0', aby serwer był dostępny dla innych procesów w sieci lokalnej
    app.run(host='0.0.0.0', port=5000, debug=True)