import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Optional: Enable Mixed Precision (if GPU supports it)
# ------------------------------
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
    print("Mixed precision enabled.")
except ImportError:
    print("Mixed precision not available; skipping.")

# ------------------------------
# SET PATHS (Update as needed)
# ------------------------------
TRAIN_DIR = r"D:\AI Assitant\asl_dataset\asl_alphabet_train"
TEST_DIR = r"D:\AI Assitant\asl_dataset\asl_alphabet_test"
SAVE_DIR = r"D:\AI Assitant\trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64  # Increased batch size for faster epoch processing

# ------------------------------
# DATA AUGMENTATION & TRAINING DATA
# ------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Create and save a label encoder using the class names from the generator.
classes = list(train_generator.class_indices.keys())
label_encoder = LabelEncoder()
label_encoder.fit(classes)
joblib.dump(label_encoder, os.path.join(SAVE_DIR, "label_encoder.joblib"))
print("Label encoder saved.")

# ------------------------------
# BUILD THE MODEL
# ------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax', dtype='float32')  # Ensure output is float32
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# CALLBACKS
# ------------------------------
checkpoint = ModelCheckpoint(
    os.path.join(SAVE_DIR, "best_model.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ------------------------------
# TRAIN THE MODEL
# ------------------------------
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ------------------------------
# EVALUATE THE MODEL ON TEST DATA
# ------------------------------
def load_test_data(test_dir, image_size):
    images = []
    labels = []
    # Test folder contains files like "A_test.jpg", "B_test.jpg", etc.
    for filename in os.listdir(test_dir):
        if "_" in filename:
            letter = filename.split("_")[0].upper()
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                img_path = os.path.join(test_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(letter)
    return np.array(images, dtype="float32") / 255.0, np.array(labels)

print("Loading test data...")
test_images, test_labels = load_test_data(TEST_DIR, IMAGE_SIZE)
print(f"Loaded {len(test_images)} test images.")

y_test_enc = label_encoder.transform(test_labels)
y_test_cat = to_categorical(y_test_enc, num_classes=26)

loss, accuracy = model.evaluate(test_images, y_test_cat, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ------------------------------
# SAVE THE FINAL MODEL AND LABEL ENCODER
# ------------------------------
model.save(os.path.join(SAVE_DIR, "best_model_final.h5"))
joblib.dump(label_encoder, os.path.join(SAVE_DIR, "label_encoder.joblib"))
print(f"Final model saved as {os.path.join(SAVE_DIR, 'best_model_final.h5')}")
print(f"Label encoder saved as {os.path.join(SAVE_DIR, 'label_encoder.joblib')}")
print("Training and testing complete.")
