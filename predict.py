import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = tf.keras.models.load_model("brain_tumor_cnn_model.keras")

# Class names (MUST match training folder names)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Folder containing test images
TEST_FOLDER = r"C:\python projects\test_folder"

print("🔍 Predicting images from:", TEST_FOLDER)
print("------------------------------------------------")

for img_name in os.listdir(TEST_FOLDER):
    img_path = os.path.join(TEST_FOLDER, img_name)

    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"🖼 {img_name} → {predicted_class} ({confidence:.2f}%)")
