import cv2
import easyocr
import re
import os
import numpy as np

# Inicjalizacja OCR
reader = easyocr.Reader(['pl'])

# Foldery
INPUT_FOLDER = 'zapisane_tablice'
OUTPUT_FOLDER = 'wyniki'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Obsługiwane rozszerzenia
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def preprocess_variants(image):
    variants = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(('oryginal', image))
    variants.append(('szarosć', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
    variants.append(('kontrast', cv2.cvtColor(cv2.convertScaleAbs(gray, alpha=1.5, beta=0), cv2.COLOR_GRAY2BGR)))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(('clahe', cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)))
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(('threshold', cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)))
    scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(('skalowanie×2', scaled))
    return variants

def perform_ocr_and_merge(img):
    raw = reader.readtext(img, decoder='wordbeamsearch', beamWidth=15)
    fragments = []
    for bbox, text, conf in raw:
        if conf >= 0.1:
            cleaned = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
            if cleaned:
                fragments.append((bbox, cleaned, conf))
    if not fragments:
        return None
    fragments.sort(key=lambda x: x[0][0][0])
    full = ''.join([f[1] for f in fragments])
    avg_conf = sum(f[2] for f in fragments) / len(fragments)
    return full, avg_conf, fragments

# Główna pętla
for fname in os.listdir(INPUT_FOLDER):
    if not fname.lower().endswith(VALID_EXTS):
        continue

    path = os.path.join(INPUT_FOLDER, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Nie wczytano {fname}")
        continue

    print(f"\n🔍 Przetwarzanie: {fname}")
    best_result = None
    best_variant_name = ''
    best_image = None

    for var_name, var_img in preprocess_variants(img):
        result = perform_ocr_and_merge(var_img)
        var_img_annotated = var_img.copy()

        if result:
            text_full, conf_avg, fragments = result
            print(f"🔎 {var_name}: {text_full} (avg_conf={conf_avg:.2f})")

            # Rysuj boxy na każdej wersji
            for bbox, txt, conf in fragments:
                tl = tuple(map(int, bbox[0]))
                br = tuple(map(int, bbox[2]))
                cv2.rectangle(var_img_annotated, tl, br, (255, 0, 0), 2)
                cv2.putText(var_img_annotated, txt, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if not best_result or conf_avg > best_result[1]:
                best_result = (text_full, conf_avg, fragments)
                best_variant_name = var_name
                best_image = var_img_annotated.copy()

        # Zapisz KAŻDY wariant, nawet jeśli nie był najlepszy
        out_variant = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(fname)[0]}_{var_name}.jpg")
        cv2.imwrite(out_variant, var_img_annotated)

    # Jeśli któryś wariant miał wynik ≥ 0.8
    if best_result and best_result[1] >= 0.8:
        text_full, conf_avg, fragments = best_result
        print(f"✅ Najlepszy wynik: {text_full} (conf={conf_avg:.2f}) przy `{best_variant_name}`")

        # Zapis obrazu z najlepszym wynikiem
        out_img = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(fname)[0]}_{best_variant_name}_najlepszy.jpg")
        cv2.imwrite(out_img, best_image)
        print(f"   📸 Zapisano obraz: {out_img}")

        # Zapis tekstu
        out_txt = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(fname)[0]}.txt")
        with open(out_txt, 'w') as f:
            f.write(f"{text_full}  (avg_conf={conf_avg:.2f})\n")
        print(f"   📝 Zapisano tekst:  {out_txt}")
    else:
        print(f"⚠️ Brak wyniku ≥ 0.8 dla {fname}")
