import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === Rutas ===
ruta_dataset = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\recortadas"
ruta_modelo = os.path.join(ruta_dataset, "modelo_binario_lick_vs_nolick.h5")

# === Parámetros ===
img_size = (128, 128)
batch_size = 32
epochs = 15

# === Preprocesamiento y augmentación de datos ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    ruta_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    classes=['Lick', 'NoLick']
)
print("📂 Asignación de clases:", train_gen.class_indices)

val_gen = datagen.flow_from_directory(
    ruta_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    classes=['Lick', 'NoLick']
)

# === Modelo CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Salida binaria
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === Entrenar ===
history=model.fit(train_gen, epochs=epochs, validation_data=val_gen)

# === Guardar modelo ===
model.save(ruta_modelo)
print(f"✅ Modelo binario guardado en: {ruta_modelo}")

# === Gráfico de precisión ===
plt.figure()
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión del modelo binario')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfico de pérdida (error) ===
plt.figure()
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida del modelo binario')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
