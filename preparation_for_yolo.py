import os

# Ścieżka do katalogu z plikami txt
folder_path = './images'  # lub np. './txt_files'

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue  # pomiń puste linie
            if parts[0] == '4':
                parts[0] = '0'
            elif parts[0] == '15':
                parts[0] = '1'
            new_lines.append(' '.join(parts) + '\n')

        # Nadpisz plik
        with open(file_path, 'w') as file:
            file.writelines(new_lines)

print("Zamiana zakończona.")
