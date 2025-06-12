import sqlite3

# Tworzymy (lub otwieramy jeśli już istnieje) plik bazy danych
conn = sqlite3.connect('plates.db')
cursor = conn.cursor()

# Tworzymy tabelę, jeśli jeszcze nie istnieje
cursor.execute('''
CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT UNIQUE NOT NULL
)
''')

conn.commit()
print("Tabela plates została utworzona lub już istnieje.")


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
        
#selectDB()

conn.close()