import os
import xml.etree.ElementTree as ET
import shutil

def convert_voc_to_yolo(xml_folder, output_txt_folder, image_folder, output_img_folder):
    os.makedirs(output_txt_folder, exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_id = 0  # Załóżmy: tylko klasa "car"

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # YOLO format: class x_center y_center width height (wszystko w [0,1])
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Zapisz plik .txt z labelami
        img_filename = root.find("filename").text
        txt_filename = img_filename.replace(".jpg", ".txt").replace(".png", ".txt")
        with open(os.path.join(output_txt_folder, txt_filename), "w") as f:
            f.write("\n".join(yolo_lines))

        # Skopiuj obraz do odpowiedniego folderu
        img_src = os.path.join(image_folder, img_filename)
        img_dst = os.path.join(output_img_folder, img_filename)
        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        else:
            print(f"❗ Brak obrazu: {img_src}")

# --- Użycie ---
print("🔄 Konwertowanie danych...")

convert_voc_to_yolo(
    xml_folder="Aerial-cars/train_PSU",
    output_txt_folder="dataset/labels/train",
    image_folder="Aerial-cars/train_PSU",
    output_img_folder="dataset/images/train"
)

convert_voc_to_yolo(
    xml_folder="Aerial-cars/test_PSU",
    output_txt_folder="dataset/labels/val",
    image_folder="Aerial-cars/test_PSU",
    output_img_folder="dataset/images/val"
)

print("✅ Gotowe! Wszystkie etykiety i obrazy skopiowane.")
