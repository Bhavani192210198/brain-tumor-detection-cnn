import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =========================
# BASIC SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# =========================
# DATASET PATH
# =========================
DATASET_PATH = r"C:\Users\91834\Downloads\brain tumor dataset\Training"

# =========================
# DATA PREPROCESSING
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # ✅ multi-class
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # ✅ multi-class
    subset='validation'
)

# Print class labels
print("Class labels:", train_data.class_indices)

# =========================
# CNN MODEL
# =========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')  # ✅ 4 tumor classes
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# =========================
# SAVE MODEL (RECOMMENDED FORMAT)
# =========================
model.save("brain_tumor_cnn_model.keras")
print("✅ Model training completed and saved successfully.")

# =========================
# OVERFITTING / UNDERFITTING CHECK
# =========================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.show()
