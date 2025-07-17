import tensorflow as tf
import numpy as np
import cv2
import os

# === RUTA DEL MODELO ===
ruta_modelo = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\recortadas\mejor_modelo_14_0.7012.h5"

# === CARGAR MODELO ===
modelo = tf.keras.models.load_model(ruta_modelo)

# === CLASES (ajusta si el orden cambia) ===
clases = ["Lick", "MedioLick", "NoLick"]

# === CARGAR Y PREPROCESAR UNA IMAGEN ===
ruta_imagen = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\recortadas\Lick\ejemplo_1.png"  # <- Cambia por la imagen que quieras probar

img = cv2.imread(ruta_imagen)
img = cv2.resize(img, (128, 128))        # Asegúrate que coincida con el tamaño usado al entrenar
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)        # Para meterlo al modelo (forma: [1, 128, 128, 3])

# === PREDICCIÓN ===
pred = modelo.predict(img)
indice_clase = np.argmax(pred)
probabilidad = pred[0][indice_clase]
clase = clases[indice_clase]

print(f"Predicción: {clase} (confianza: {probabilidad:.4f})")
