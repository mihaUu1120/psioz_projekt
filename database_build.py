import sqlite3
from datetime import datetime

# Tworzymy (lub otwieramy jeśli już istnieje) plik bazy danych
conn = sqlite3.connect('plates.db')
cursor = conn.cursor()

# Tworzymy tabelę, jeśli jeszcze nie istnieje
cursor.execute('''
CREATE TABLE IF NOT EXISTS plates (
    plate_number TEXT PRIMARY KEY,
    x1 INTEGER,
    y1 INTEGER,
    x2 INTEGER,
    y2 INTEGER,
    last_update TEXT
)
''')
conn.commit()
print("Tabela plates została utworzona lub już istnieje.")

def is_plate_in_db(plate):
    cursor.execute("SELECT 1 FROM plates WHERE plate_number = ?", (plate,))
    return cursor.fetchone() is not None

def add_plate_to_db(plate, x1=None, y1=None, x2=None, y2=None):
    if not is_plate_in_db(plate):
        cursor.execute(
            "INSERT INTO plates (plate_number, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?)",
            (plate, x1, y1, x2, y2)
        )
        conn.commit()
        print(f"Dodano tablicę '{plate}' do bazy danych.")

def update_plate_position(plate, x1, y1, x2, y2):
    now = datetime.now().isoformat()
    cursor.execute('''
        UPDATE plates SET x1 = ?, y1 = ?, x2 = ?, y2 = ?, last_update = ?
        WHERE plate_number = ?
    ''', (x1, y1, x2, y2, now, plate))
    conn.commit()
    print(f"Zaktualizowano pozycję tablicy '{plate}'.")
    

def add_plate(plate):
    try:
        cursor.execute("INSERT INTO plates (plate_number) VALUES (?)", (plate,))
        conn.commit()
        print(f"Dodano tablicę: {plate}")
    except sqlite3.IntegrityError:
        print(f"Tablica {plate} już istnieje w bazie.")

# Przykład dodania kilku tablic


# add_plate("2115")

def selectDB():
    cursor.execute("SELECT * FROM plates")
    rows = cursor.fetchall()

    print("Zawartość bazy plates:")
    for row in rows:
        print(f"ID: {row[0]}, Tablica: {row[1]}")
        
# selectDB()

# cursor.execute("DELETE FROM plates")
# conn.commit()

conn.close()