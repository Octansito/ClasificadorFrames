import os
import cv2
import re
import random

# === Configuración de rutas ===
ruta_base = r"C:\MisArchivos\Escritorio\ClasificadorFrames"
ruta_videos = os.path.join(ruta_base, "videos")
ruta_txts = os.path.join(ruta_base, "anotaciones", "txtPruebas")
out_extra = os.path.join(ruta_base, "dataset", "FramesExtraidos")
os.makedirs(out_extra, exist_ok=True)

# === Obtener lista de videos .avi ===
videos = [f for f in os.listdir(ruta_videos) if f.lower().endswith(".avi")]
if not videos:
    print("❌ No se encontraron videos .avi en la carpeta.")
    exit()

# === Procesar cada video ===
for video_name in videos:
    video_path = os.path.join(ruta_videos, video_name)
    nombre_base = os.path.splitext(video_name)[0]

    # Extraer fecha del nombre del video
    match = re.match(r"(\d{2})-(\d{2})-(\d{2})", nombre_base)
    if not match:
        print(f"⚠️ No se pudo extraer fecha de {video_name}, se omite.")
        continue

    dd, mm, yy = match.groups()
    fecha_txt = yy + mm + dd

    # Buscar archivo .txt que contenga esa fecha y termine en _eventframe
    posibles_txts = [
        f for f in os.listdir(ruta_txts)
        if fecha_txt in f and "eventframe" in f
    ]
    if not posibles_txts:
        print(f"⚠️ No se encontró .txt para {video_name}, se omite.")
        continue

    txt_path = os.path.join(ruta_txts, posibles_txts[0])
    print(f"🔍 Procesando {video_name} con anotaciones: {posibles_txts[0]}")

    # Leer lista de lick frames
    lick_frames_all = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                lick_frames_all.append(int(line))

    if len(lick_frames_all) < 15:
        print(f"⚠️ No hay suficientes frames en {txt_path}, se omite.")
        continue

    lick_frames_all = sorted(lick_frames_all)

    #Cuando haya mas licks, recortar a 15
    # Dividir los frames en 50 bloques equidistantes
    num_licks=50
    bloque_size = len(lick_frames_all) // 50
    lick_frames_distribuidos = []
    for i in range(num_licks):
        start_idx = i * bloque_size
        end_idx = (i + 1) * bloque_size if i < 14 else len(lick_frames_all)
        bloque = lick_frames_all[start_idx:end_idx]
        if bloque:
            lick_frames_distribuidos.append(random.choice(bloque))

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    guardados = 0

    for frame_central in lick_frames_distribuidos:
        for offset in range(-50, 51):  # de -10 a +10
            actual = frame_central + offset
            if 0 <= actual < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual)
                ret, frame = cap.read()
                if not ret:
                    continue
                nombre_frame = f"{nombre_base}_from_{frame_central}_frame_{actual}.png"
                salida = os.path.join(out_extra, nombre_frame)
                cv2.imwrite(salida, frame)
                guardados += 1

    cap.release()
    print(f"✅ {video_name}: Se guardaron {guardados} frames alrededor de 15 licks distribuidos\n")
