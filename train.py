import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración de parámetros
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 8
DATA_DIR_TRAIN = "train"
DATA_DIR_TEST = "test"
MODEL_PATH = "mejor_modelo.h5"
PLOT_PATH = "grafico_entrenamiento.png"

# Paso 1: Data Augmentation (Aumento de datos)
print("🔄 Configurando Data Augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Cargar datos desde directorios
print("📂 Cargando datos de entrenamiento...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=True,
)

print("📂 Cargando datos de prueba...")
test_generator = test_datagen.flow_from_directory(
    DATA_DIR_TEST,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False,
)

# Obtener número de clases
num_classes = len(train_generator.class_indices)
emociones = list(train_generator.class_indices.keys())
print(f"✅ Clases detectadas: {emociones}")

# Paso 2: Definir un modelo CNN más robusto para FER-2013
print("🧠 Construyendo modelo CNN optimizado para expresiones faciales...")
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.30),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.35),

    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation="softmax"),
])

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Paso 3: Callbacks
print("⚙️ Configurando callbacks...")
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    mode="max",
    verbose=1,
)

model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    mode="max",
    verbose=1,
)

# Balancear clases ayuda bastante en FER, especialmente "disgust"
class_counts = np.bincount(train_generator.classes)
total_samples = np.sum(class_counts)
class_weights = {
    class_index: float(total_samples / (len(class_counts) * count))
    for class_index, count in enumerate(class_counts)
    if count > 0
}
print(f"⚖️ Class weights: {class_weights}")

# Paso 4: Entrenar el modelo
print("🚀 Iniciando entrenamiento...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    class_weight=class_weights,
    verbose=1,
)

# Paso 5: Mostrar gráfico de precisión
print("📊 Generando gráfico de precisión...")
plt.figure(figsize=(12, 4))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print(f"✅ Entrenamiento completado! Modelo guardado como '{MODEL_PATH}'")
