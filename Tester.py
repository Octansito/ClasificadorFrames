import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Ruta al modelo binario ===
modelo_path = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\modelo_binario_lick_vs_nolick.h5"
model = load_model(modelo_path)

# === Imagen a probar ===
img_path = r"C:\MisArchivos\Escritorio\ClasificadorFrames\resultados_binario\24-10-23 12-52-35\NoLick\frame_0196.png"
img = cv2.imread(img_path)

# === Recorte si hace falta (ya está recortada en este caso)
img = cv2.resize(img, (128, 128))  # igual que entrenamiento
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Predicción
pred = model.predict(img_array)[0][0]
print(f"Predicción para esta imagen: {pred:.4f} ({'Lick' if pred >= 0.5 else 'NoLick'})")
