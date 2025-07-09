import os
import shutil
from collections import defaultdict

# === CONFIGURACIÓN DE RUTAS ===
origen = r"C:\MisArchivos\Escritorio\ClasificadorFrames\anotaciones\Batch2"
destino = r"C:\MisArchivos\Escritorio\ClasificadorFrames\anotaciones\NuevosTxtBatch2"
log_path = os.path.join(destino, "log_renombrado.txt")

# Crear carpeta destino si no existe
os.makedirs(destino, exist_ok=True)

# Obtener lista de archivos .txt ordenados por nombre
archivos = sorted([f for f in os.listdir(origen) if f.endswith(".txt")])

# Inicialización
letras_por_fecha = defaultdict(int)
log_lines = []
ultima_fecha = ""

# Procesar archivos
for archivo in archivos:
    nombre_sin_ext = archivo.replace(".txt", "")
    partes = nombre_sin_ext.split("_")

    if len(partes) != 2:
        log_lines.append(f"❌ Ignorado por formato inesperado: {archivo}")
        continue

    fecha, hora = partes  # ejemplo: 02-12-24 y 09-42-18
    dia = fecha

    if dia != ultima_fecha:
        letras_por_fecha[dia] = 0
        ultima_fecha = dia

    letra_idx = letras_por_fecha[dia]
    if letra_idx >= 26:
        log_lines.append(f"❌ Demasiados archivos para {dia}, ignorando {archivo}")
        continue

    letra = chr(ord('A') + letra_idx)
    letras_por_fecha[dia] += 1

    dd, mm, yy = dia.split("-")
    nuevo_nombre = f"{letra}{dd}{mm}{yy}_eventframe.txt"

    ruta_origen = os.path.join(origen, archivo)
    ruta_destino = os.path.join(destino, nuevo_nombre)

    if os.path.exists(ruta_destino):
        log_lines.append(f"⚠️ Ya existe: {nuevo_nombre}, saltando {archivo}")
        continue

    shutil.copy2(ruta_origen, ruta_destino)
    log_lines.append(f"✅ {archivo} -> {nuevo_nombre}")

# Guardar log
with open(log_path, "w", encoding="utf-8") as log_file:
    log_file.write("\n".join(log_lines))

print(f"📄 Proceso finalizado. Log guardado en:\n{log_path}")
