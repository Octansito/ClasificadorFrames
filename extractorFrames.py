import os
import cv2
import random

# === Configuración de rutas ===
ruta_base = r"C:\MisArchivos\Escritorio\ClasificadorFrames"
ruta_videos = os.path.join(ruta_base, "videos")
ruta_txts = os.path.join(ruta_base, "anotaciones")
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

    # Buscar archivo .txt correspondiente que empiece igual
    posibles_txts = [
        f for f in os.listdir(ruta_txts)
        if f.startswith(nombre_base) and f.endswith("_filtered_events_groundtruth.txt")
    ]
    if not posibles_txts:
        print(f"⚠️ No se encontró .txt para {video_name}, se omite.")
        continue

    txt_path = os.path.join(ruta_txts, posibles_txts[0])

    print(f"🔍 Procesando {video_name} con anotaciones: {posibles_txts[0]}")

    # Leer lista de lick frames
    with open(txt_path, "r") as f:
        lick_frames_all = sorted(set(int(line.strip()) for line in f if line.strip().isdigit()))
        #Leemos los frames del txt
    with open(txt_path,"r") as f:
        lick_frames_all=sorted(set(int(line.strip()) for line in f if line.strip().isdigit()))
    
    #Elegir 15 licks aleatorios de todo el rango
    if len(lick_frames_all)>=15:
        lick_frames=sorted(random.sample(lick_frames_all,15))
    else:
        lick_frames=lick_frames_all
            
    # Abrir video
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
