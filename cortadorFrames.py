import os
import cv2

# === Rutas base ===
ruta_base = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset"
output_base = os.path.join(ruta_base, "recortadas")

# === Coordenadas centradas para 720x480 con recorte 408x333 ===
crop_box = (50, 73, 500, 333)  # (x, y, ancho, alto)

# === Recorrer cada clase ===
for clase in ["Lick", "NoLick", "MedioLick"]:
    entrada_clase = os.path.join(ruta_base, clase)
    salida_clase = os.path.join(output_base, clase)
    os.makedirs(salida_clase, exist_ok=True)

    count = 0
    for filename in os.listdir(entrada_clase):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(entrada_clase, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            x, y, w, h = crop_box
            recorte = img[y:y+h, x:x+w]
            out_path = os.path.join(salida_clase, filename)
            cv2.imwrite(out_path, recorte)
            count += 1

    print(f"✅ Recortadas {count} imágenes en '{clase}'")
