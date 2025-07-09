import os
import cv2
import re

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

    if len(lick_frames_all) < 2:
        print(f"⚠️ No hay suficientes frames en {txt_path}, se omite.")
        continue

    lick_frames_all = sorted(lick_frames_all)

    # Buscar dos frames con diferencia ≥ 3000
    found_pair = False
    for i in range(len(lick_frames_all)):
        for j in range(i + 1, len(lick_frames_all)):
            if abs(lick_frames_all[j] - lick_frames_all[i]) >= 20000:
                start_frame = lick_frames_all[i]
                end_frame = lick_frames_all[j]
                found_pair = True
                break
        if found_pair:
            break

    if not found_pair:
        print(f"⚠️ No se encontraron dos frames con separación ≥ 15000 en {txt_path}")
        continue

    print(f"🎯 Rango seleccionado: {start_frame} - {end_frame} ({end_frame - start_frame + 1} frames)")

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Asegurar que están dentro del rango del video
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames - 1)

    # Extraer frames en el rango
    guardados = 0
    for actual in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual)
        ret, frame = cap.read()
        if not ret:
            continue

        nombre_frame = f"{nombre_base}_frame_{actual}.png"
        salida = os.path.join(out_extra, nombre_frame)
        cv2.imwrite(salida, frame)
        guardados += 1

    cap.release()
    print(f"✅ {video_name}: Se guardaron {guardados} frames en FramesExtraidos\n")
