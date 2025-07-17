import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import defaultdict
from tqdm import tqdm

# === Rutas ===
ruta_videos = r"C:\MisArchivos\Escritorio\ClasificadorFrames\videos"
modelo_binario = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\modelo_binario_lick_vs_nolick.h5"
salida_base = r"C:\MisArchivos\Escritorio\ClasificadorFrames\resultados_binario"
os.makedirs(salida_base, exist_ok=True)

# === Cargar modelo ===
model = load_model(modelo_binario)

# === Coordenadas de recorte ===
x, y, w, h = 50, 73, 500, 333  # Ajustadas según recorte

# === Procesar videos ===
for video_file in os.listdir(ruta_videos):
    if not video_file.endswith('.avi'):
        continue

    nombre_video = os.path.splitext(video_file)[0]
    path_video = os.path.join(ruta_videos, video_file)
    cap = cv2.VideoCapture(path_video)

    salida_txt = os.path.join(salida_base, f"{nombre_video}_clasificacion.txt")
    carpeta_frames = os.path.join(salida_base, nombre_video)
    os.makedirs(carpeta_frames, exist_ok=True)

    carpetas = {
        'Lick': os.path.join(carpeta_frames, 'Lick'),
        'NoLick': os.path.join(carpeta_frames, 'NoLick')
    }
    for carpeta in carpetas.values():
        os.makedirs(carpeta, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clasificacion = defaultdict(list)

    frame_id = 0
    with tqdm(total=total_frames, desc=f"Analizando {video_file}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recorte
            recorte = frame[y:y+h, x:x+w]
            recorte = cv2.resize(recorte, (128, 128))
            img_array = img_to_array(recorte) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predicción
            pred = model.predict(img_array, verbose=0)[0][0]
            label = 'NoLick' if pred >= 0.5 else 'Lick'  #Ponerlo al revés
            clasificacion[label].append(frame_id)

            out_path = os.path.join(carpetas[label], f"frame_{frame_id:04d}.png")
            #cv2.imwrite(out_path, recorte)  evitamos que se copien los frames

        frame_id += 1
        pbar.update(1)

    cap.release()

        # === Guardar resultados ===
    total = frame_id
    lick_pct = len(clasificacion['Lick']) / total * 100
    nolick_pct = len(clasificacion['NoLick']) / total * 100

    resumen = []
    resumen.append("Resumen:\n")
    resumen.append(f"Total frames: {total}\n")
    resumen.append(f"Lick: {len(clasificacion['Lick'])} ({lick_pct:.2f}%)\n")
    resumen.append(f"NoLick: {len(clasificacion['NoLick'])} ({nolick_pct:.2f}%)\n\n")

    tabla = []
    tabla.append("Frames clasificados por tipo:\n")
    tabla.append("Lick       | NoLick\n")
    tabla.append("-----------------------\n")
    max_len = max(len(clasificacion['Lick']), len(clasificacion['NoLick']))
    for i in range(max_len):
        lick_f = clasificacion['Lick'][i] if i < len(clasificacion['Lick']) else ''
        nolick_f = clasificacion['NoLick'][i] if i < len(clasificacion['NoLick']) else ''
        tabla.append(f"{str(lick_f):<11}| {str(nolick_f):<}\n")

    with open(salida_txt, 'w') as f:
        f.writelines(resumen + tabla)


print("✅ Clasificación binaria completada.")
