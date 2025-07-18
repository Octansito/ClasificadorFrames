import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import defaultdict
from tqdm import tqdm

# === Rutas ===
ruta_videos = r"C:\MisArchivos\Escritorio\ClasificadorFrames\videos"
modelo_multiclase = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\modelo_multiclase_lick_medionolick_FINAL_70.h5"
salida_base = r"C:\MisArchivos\Escritorio\ClasificadorFrames\resultados_multiclase"
os.makedirs(salida_base, exist_ok=True)

# === Cargar modelo ===
model = load_model(modelo_multiclase)
clases = ['Lick', 'MedioLick', 'NoLick']

# === Coordenadas de recorte ===
x, y, w, h = 50, 73, 500, 333  # Ajustadas según recorte

# === Parámetros ===
BATCH_SIZE = 64

# === Procesar videos ===
for video_file in os.listdir(ruta_videos):
    if not video_file.endswith('.avi'):
        continue

    nombre_video = os.path.splitext(video_file)[0]
    path_video = os.path.join(ruta_videos, video_file)
    cap = cv2.VideoCapture(path_video)

    salida_txt = os.path.join(salida_base, f"{nombre_video}_clasificacion.txt")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clasificacion = defaultdict(list)

    frame_id = 0
    batch_imgs = []
    batch_ids = []

    with tqdm(total=total_frames, desc=f"Analizando {video_file}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            recorte = frame[y:y+h, x:x+w]
            recorte = cv2.resize(recorte, (128, 128))
            img_array = img_to_array(recorte) / 255.0
            batch_imgs.append(img_array)
            batch_ids.append(frame_id)

            if len(batch_imgs) == BATCH_SIZE:
                batch_array = np.array(batch_imgs)
                preds = model.predict(batch_array, verbose=0)

                for pred, fid in zip(preds, batch_ids):
                    clase_idx = np.argmax(pred)
                    label = clases[clase_idx]
                    clasificacion[label].append(fid)

                batch_imgs.clear()
                batch_ids.clear()

            frame_id += 1
            pbar.update(1)

    # Últimos frames si el batch no está completo
    if batch_imgs:
        batch_array = np.array(batch_imgs)
        preds = model.predict(batch_array, verbose=0)
        for pred, fid in zip(preds, batch_ids):
            clase_idx = np.argmax(pred)
            label = clases[clase_idx]
            clasificacion[label].append(fid)

    cap.release()

    # === Guardar resultados ===
    total = frame_id
    resumen = ["Resumen:\n"]
    for c in clases:
        porcentaje = len(clasificacion[c]) / total * 100
        resumen.append(f"{c}: {len(clasificacion[c])} ({porcentaje:.2f}%)\n")
    resumen.append(f"Total frames: {total}\n\n")

    tabla = ["Frames clasificados por tipo:\n"]
    tabla.append("Lick       | MedioLick  | NoLick\n")
    tabla.append("-------------------------------------\n")
    max_len = max(len(clasificacion['Lick']), len(clasificacion['MedioLick']), len(clasificacion['NoLick']))
    for i in range(max_len):
        lick = clasificacion['Lick'][i] if i < len(clasificacion['Lick']) else ''
        medio = clasificacion['MedioLick'][i] if i < len(clasificacion['MedioLick']) else ''
        nolick = clasificacion['NoLick'][i] if i < len(clasificacion['NoLick']) else ''
        tabla.append(f"{str(lick):<11}| {str(medio):<11}| {str(nolick):<}\n")

    with open(salida_txt, 'w') as f:
        f.writelines(resumen + tabla)

print("✅ Clasificación multiclase completada.")
