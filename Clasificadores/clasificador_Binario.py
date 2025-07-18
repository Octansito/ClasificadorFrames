import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import defaultdict
from tqdm import tqdm
#Análisis hecho por batch de 32
# === Rutas ===
ruta_videos = r"C:\MisArchivos\Escritorio\ClasificadorFrames\videos"
modelo_binario = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\modelo_binario_lick_vs_nolick.h5"
salida_base = r"C:\MisArchivos\Escritorio\ClasificadorFrames\resultados_binario"
os.makedirs(salida_base, exist_ok=True)

# === Cargar modelo ===
model = load_model(modelo_binario)

# === Coordenadas de recorte ===
x, y, w, h = 50, 73, 500, 333

# === Procesar videos ===
batch_size = 64

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

            frame_id += 1
            pbar.update(1)

            # Si se completa un batch
            if len(batch_imgs) == batch_size:
                batch_array = np.array(batch_imgs)
                preds = model.predict(batch_array, verbose=0).flatten()

                for i, pred in enumerate(preds):
                    label = 'NoLick' if pred >= 0.5 else 'Lick'
                    clasificacion[label].append(batch_ids[i])

                batch_imgs = []
                batch_ids = []

        # Procesar cualquier lote restante
        if batch_imgs:
            batch_array = np.array(batch_imgs)
            preds = model.predict(batch_array, verbose=0).flatten()

            for i, pred in enumerate(preds):
                label = 'NoLick' if pred >= 0.5 else 'Lick'
                clasificacion[label].append(batch_ids[i])

    cap.release()

    # === Guardar resultados ===
    total = frame_id
    lick_pct = len(clasificacion['Lick']) / total * 100
    nolick_pct = len(clasificacion['NoLick']) / total * 100

    resumen = [
        "Resumen:\n",
        f"Total frames: {total}\n",
        f"Lick: {len(clasificacion['Lick'])} ({lick_pct:.2f}%)\n",
        f"NoLick: {len(clasificacion['NoLick'])} ({nolick_pct:.2f}%)\n\n"
    ]

    tabla = [
        "Frames clasificados por tipo:\n",
        "Lick       | NoLick\n",
        "-----------------------\n"
    ]

    max_len = max(len(clasificacion['Lick']), len(clasificacion['NoLick']))
    for i in range(max_len):
        lick_f = clasificacion['Lick'][i] if i < len(clasificacion['Lick']) else ''
        nolick_f = clasificacion['NoLick'][i] if i < len(clasificacion['NoLick']) else ''
        tabla.append(f"{str(lick_f):<11}| {str(nolick_f):<}\n")

    with open(salida_txt, 'w') as f:
        f.writelines(resumen + tabla)

print("✅ Clasificación binaria completada.")
