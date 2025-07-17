import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# === Rutas ===
ruta_dataset = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\recortadas"
ruta_modelo_final = os.path.join(ruta_dataset, "modelo_multiclase_lick_medionolick_FINAL.h5")
ruta_guardado_top = os.path.join(ruta_dataset, "mejores_modelos")

# === Crear carpeta para guardar los mejores modelos
os.makedirs(ruta_guardado_top, exist_ok=True)

# === Parámetros ===
img_size = (128, 128)
batch_size = 32
epochs = 15

# === Preprocesamiento ===
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
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    ruta_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
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
    Dense(4, activation='softmax')  # Salida multiclase
])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === Callback para guardar los 5 mejores modelos ===
ruta_checkpoint = os.path.join(ruta_guardado_top, "mejor_modelo_{epoch:02d}_{val_accuracy:.4f}.h5")
checkpoint = ModelCheckpoint(
    ruta_checkpoint,
    monitor='val_accuracy',
    save_best_only=False,
    save_weights_only=False,
    verbose=1,
    save_freq='epoch'
)

# === Entrenamiento ===
hist = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[checkpoint]
)

# === Guardar modelo final ===
model.save(ruta_modelo_final)
print(f"✅ Modelo final multiclase guardado en: {ruta_modelo_final}")

# === Gráficas ===
plt.plot(hist.history['accuracy'], label='Precisión entrenamiento')
plt.plot(hist.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión del modelo multiclase')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid()
plt.show()

plt.plot(hist.history['loss'], label='Pérdida entrenamiento')
plt.plot(hist.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida del modelo multiclase')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid()
plt.show()
