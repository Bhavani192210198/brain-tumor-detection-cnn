import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

# 🔹 DATA PREPROCESSING
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# 🔹 UPDATE THIS PATH (VERY IMPORTANT)
DATASET_PATH = r"C:\Users\91834\Downloads\brain tumor dataset\Training"

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # 🔴 changed from binary
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # 🔴 changed from binary
    subset='validation'
)

# 🔹 PRINT CLASSES (OPTIONAL BUT GOOD)
print("Class labels:", train_data.class_indices)

# 🔹 CNN MODEL
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

    Dense(4, activation='softmax')  # 🔴 4 classes
])

# 🔹 COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # 🔴 changed loss
    metrics=['accuracy']
)

# 🔹 TRAIN MODEL
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# 🔹 SAVE MODEL
model.save("brain_tumor_cnn_model.h5")

print("✅ Model training completed and saved successfully.")
