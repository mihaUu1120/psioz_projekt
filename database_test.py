import sqlite3

# Połączenie z bazą danych (utworzy plik, jeśli nie istnieje)
conn = sqlite3.connect('parking.db')
cursor = conn.cursor()

def add_allowed_plate_to_db(plate):
    cursor.execute(
        "INSERT INTO allowed_plates (plate_number) VALUES (?)",
        (plate,)
    )
    conn.commit()
    print(f"Dodano dozwoloną tablicę '{plate}' do bazy danych.")


def display_table(table):
    allowed_tables = ['entries_exits', 'plates', 'allowed_plates', 'forbidden_moves']

    if table not in allowed_tables:
        raise ValueError("Niedozwolona nazwa tabeli.")

    query = f'SELECT * FROM {table}'
    cursor.execute(query)
    rows = cursor.fetchall()

    for row in rows:
        print(row)


def delete_from_table(table):
    allowed_tables = ['entries_exits', 'plates', 'allowed_plates', 'forbidden_moves']

    if table not in allowed_tables:
        raise ValueError("Niedozwolona nazwa tabeli.")
    
    query = f'DELETE FROM {table}'
    cursor.execute(query)
    conn.commit()
    print(f"Usunięto wszystkie dane z tabeli '{table}'")

def clear_all_temp_data():
    allowed_tables = ['entries_exits', 'plates', 'forbidden_moves']

    for table in allowed_tables:
        query = f'DELETE FROM {table}'
        cursor.execute(query)
    conn.commit()
    print(f"Usunięto wszystkie dane z tabel typu temp")

def list_tables():
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        print(table[0])


cursor.execute('''
CREATE TABLE IF NOT EXISTS plates (
    plate_number TEXT PRIMARY KEY,
    x1 INTEGER,
    y1 INTEGER,
    x2 INTEGER,
    y2 INTEGER,
    last_update TEXT DEFAULT CURRENT_TIMESTAMP
)
''')

# Tworzenie nowej tabeli entries_exits
cursor.execute('''
CREATE TABLE IF NOT EXISTS entries_exits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP
)
''')

# Tworzenie nowej tabeli allowed_plates
cursor.execute('''
CREATE TABLE IF NOT EXISTS allowed_plates (
    plate_number TEXT PRIMARY KEY
)
''')

# Tworzenie nowej tabeli forbidden_moves
cursor.execute('''
CREATE TABLE IF NOT EXISTS forbidden_moves (
    plate_number TEXT PRIMARY KEY,
    forbidden_time TIMESTAMP,
    type TEXT
)
''')

conn.commit()


# add_allowed_plate_to_db("8008")
# add_allowed_plate_to_db("7007")
# list_tables()
# display_table("forbidden_moves")
display_table("entries_exits")
# display_table("allowed_plates")
# clear_all_temp_data()
# delete_from_table("forbidden_moves")
# display_table("plates")

conn.close()
