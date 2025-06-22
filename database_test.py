import sqlite3

# Połączenie z bazą danych (utworzy plik, jeśli nie istnieje)
conn = sqlite3.connect('parking.db')
cursor = conn.cursor()




# SELECT * — pobranie i wyświetlenie wszystkich danych z tabeli
cursor.execute('SELECT * FROM entries_exits')
rows = cursor.fetchall()

for row in rows:
    print(row)

# Zamknięcie połączenia
conn.close()
