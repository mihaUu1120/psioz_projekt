import sqlite3
import time

class ParkingDatabaseManager:
    def __init__(self, db_name='parking.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_tables()
        print(f"Database '{db_name}' connected and tables are ready.")

    def _create_tables(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            plate_number TEXT PRIMARY KEY,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            last_update TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries_exits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP
        )
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS allowed_plates (
            plate_number TEXT PRIMARY KEY
        )
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS forbidden_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            forbidden_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            type TEXT NOT NULL
        )
        ''')
        self.conn.commit()

    def add_entry(self, plate: str):
        self.cursor.execute(
            "INSERT INTO entries_exits (plate_number, entry_time) VALUES (?, CURRENT_TIMESTAMP)",
            (plate,)
        )
        self.conn.commit()
        print(f"Car: '{plate}' entered the parking lot.")

    def update_exit(self, plate: str):
        self.cursor.execute(
            "UPDATE entries_exits SET exit_time = CURRENT_TIMESTAMP WHERE plate_number = ? AND exit_time IS NULL",
            (plate,)
        )
        self.conn.commit()
        print(f"Car: '{plate}' exited the parking lot.")

    def add_allowed_plate(self, plate: str):
        self.cursor.execute(
            "INSERT OR IGNORE INTO allowed_plates (plate_number) VALUES (?)",
            (plate,)
        )
        self.conn.commit()
        print(f"Allowed plate '{plate}' added to the database.")

    def is_allowed_plate(self, plate: str) -> bool:
        self.cursor.execute("SELECT 1 FROM allowed_plates WHERE plate_number = ?", (plate,))
        return self.cursor.fetchone() is not None

    def is_plate_in_active_parking(self, plate: str) -> bool:
        self.cursor.execute("SELECT 1 FROM plates WHERE plate_number = ?", (plate,))
        return self.cursor.fetchone() is not None

    def add_plate_to_active_parking(self, plate: str, x1: int, y1: int, x2: int, y2: int):
        if not self.is_plate_in_active_parking(plate):
            self.cursor.execute(
                "INSERT INTO plates (plate_number, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?)",
                (plate, x1, y1, x2, y2)
            )
            self.conn.commit()
            print(f"Plate '{plate}' added to active parking in DB.")

    def delete_plate_from_active_parking(self, plate: str):
        if self.is_plate_in_active_parking(plate):
            self.cursor.execute(
                "DELETE FROM plates WHERE plate_number = ?",
                (plate,)
            )
            self.conn.commit()
            print(f"Plate '{plate}' removed from active parking in DB.")

    def update_plate_position(self, plate: str, x1: int, y1: int, x2: int, y2: int):
        self.cursor.execute('''
            UPDATE plates SET x1 = ?, y1 = ?, x2 = ?, y2 = ?, last_update = CURRENT_TIMESTAMP
            WHERE plate_number = ?
        ''', (x1, y1, x2, y2, plate))
        self.conn.commit()

    def add_forbidden_move(self, plate: str, move_type: str):
        self.cursor.execute(
            "INSERT INTO forbidden_moves (plate_number, forbidden_time, type) VALUES (?, CURRENT_TIMESTAMP, ?)",
            (plate, move_type)
        )
        self.conn.commit()
        print(f"Forbidden move '{move_type}' recorded for plate '{plate}'.")

    def load_vehicles_from_active_parking(self) -> list:
        self.cursor.execute("SELECT plate_number, x1, y1, x2, y2 FROM plates")
        known_vehicles = []
        for row in self.cursor.fetchall():
            plate, x1, y1, x2, y2 = row
            if all(v is not None for v in [x1, y1, x2, y2]):
                centroid = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
                known_vehicles.append({'plate': plate, 'centroid': centroid})
        print(f"Loaded {len(known_vehicles)} known vehicles from active parking in DB.")
        return known_vehicles

    def close(self):
        self.conn.close()
        print("Database connection closed.")
