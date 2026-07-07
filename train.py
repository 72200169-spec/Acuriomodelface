import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Configuración de parámetros
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 3  # Reducido para que termine más rápido
DATA_DIR_TRAIN = 'train'
DATA_DIR_TEST = 'test'

# Paso 1: Data Augmentation (Aumento de datos)
print("🔄 Configurando Data Augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos desde directorios
print("📂 Cargando datos de entrenamiento...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

print("📂 Cargando datos de prueba...")
test_generator = test_datagen.flow_from_directory(
    DATA_DIR_TEST,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

# Obtener número de clases
num_classes = len(train_generator.class_indices)
emociones = list(train_generator.class_indices.keys())
print(f"✅ Clases detectadas: {emociones}")

# Paso 2: Definir arquitectura CNN moderna (simplificada para entrenamiento rápido)
print("🧠 Construyendo modelo CNN...")
model = Sequential([
    # Bloque 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    
    # Bloque 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Bloque 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Bloque completamente conectado
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Paso 3: Callbacks
print("⚙️ Configurando callbacks...")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'mejor_modelo.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Paso 4: Entrenar el modelo
print("🚀 Iniciando entrenamiento...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
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
plt.savefig('grafico_entrenamiento.png')
plt.show()

print("✅ Entrenamiento completado! Modelo guardado como 'mejor_modelo.keras'")
