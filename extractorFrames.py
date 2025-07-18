import os
import cv2
import random
import re

# === Configuración de rutas ===
ruta_base = r"C:\MisArchivos\Escritorio\ClasificadorFrames"
ruta_videos = os.path.join(ruta_base, "videos")
ruta_txts = os.path.join(ruta_base, "anotaciones", "txtPruebas")
out_lick = os.path.join(ruta_base, "dataset", "Lick")
out_extra = os.path.join(ruta_base, "dataset", "FramesExtraidos")
os.makedirs(out_lick, exist_ok=True)
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

    if not lick_frames_all:
        print(f"⚠️ No se encontraron frames válidos en {txt_path}, se omite.")
        continue


    # Elegir hasta 10 licks aleatorios
    if len(lick_frames_all) >= 10:
        lick_frames = sorted(random.sample(lick_frames_all, 10))
    else:
        lick_frames = lick_frames_all

    print(f"🧪 Frames seleccionados del .txt ({len(lick_frames)}): {lick_frames}")

    # === Abrir el video y obtener total de frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    guardados = {"Lick": 0, "Extra": 0}

    for frame_id in lick_frames:
        for offset in range(-3, 4):
            actual = frame_id + offset
            if actual < 0 or actual >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, actual)
            ret, frame = cap.read()
            if not ret:
                continue

            nombre_frame = f"{nombre_base}_frame_{actual}.png"

            if offset == 0:
                salida = os.path.join(out_lick, nombre_frame)
                cv2.imwrite(salida, frame)
                guardados["Lick"] += 1
            else:
                if actual not in lick_frames:
                    salida = os.path.join(out_extra, nombre_frame)
                    cv2.imwrite(salida, frame)
                    guardados["Extra"] += 1

    cap.release()
    print(f"✅ {video_name}: {guardados['Lick']} Lick | {guardados['Extra']} FramesExtraidos\n")
