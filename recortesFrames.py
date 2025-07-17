import os
import cv2

# === Rutas ===
input_dir = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\Lick"  # Carpeta con imágenes completas
output_dir = r"C:\MisArchivos\Escritorio\ClasificadorFrames\recortadas\Lick"  # Carpeta para guardar las recortadas
os.makedirs(output_dir, exist_ok=True)

# === Parámetros de recorte ===
# Coordenadas aproximadas en formato (x_inicio, y_inicio, ancho, alto)
# Puedes ajustar estas coordenadas según la forma del recorte deseado
crop_box = (150, 150, 250, 220)  # (x, y, w, h)

# === Procesar imágenes ===
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        x, y, w, h = crop_box
        cropped = img[y:y+h, x:x+w]
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, cropped)

print("✅ Recorte completado.")
