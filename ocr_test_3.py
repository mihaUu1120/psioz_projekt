import cv2
import easyocr
import os
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("Liczba wątków CPU:", torch.get_num_threads())

def read_license_plate(image_path):
    # Wczytanie obrazu
    image = cv2.imread(image_path)
    if image is None:
        print("Nie można wczytać obrazu:", image_path)
        return

    # Inicjalizacja czytnika EasyOCR (język angielski i polski)
    reader = easyocr.Reader(['en'])

    # Wykrycie i odczyt tekstu na obrazie
    results = reader.readtext(image)

    # Rysowanie prostokątów i tekstu na obrazie
    for bbox, text, confidence in results:
        top_left = tuple([int(val) for val in bbox[0]])
        bottom_right = tuple([int(val) for val in bbox[2]])
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Konwersja obrazu z BGR (OpenCV) do RGB (Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Wyświetlenie obrazu z wykrytym tekstem
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Odczytany tekst: {', '.join([text for _, text, _ in results])}")
    plt.show()

def process_images_in_folder(folder_path):
    # Sprawdzenie, czy folder istnieje
    if not os.path.exists(folder_path):
        print(f'Folder "{folder_path}" nie istnieje.')
        return

    # Pobranie wszystkich plików graficznych z folderu
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Jeśli brak plików w folderze
    if not image_files:
        print("Nie znaleziono żadnych obrazów w folderze.")
        return

    # Iteracja przez obrazy
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f'\nPrzetwarzanie obrazu: {image_path}')
        read_license_plate(image_path)

# Podaj ścieżkę do folderu z obrazami
folder_path = 'zapisane_tablice_2_przerobione'  # Zmień na ścieżkę do swojego folderu
process_images_in_folder(folder_path)
