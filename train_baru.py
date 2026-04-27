import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from asl_modules import (
    extract_landmarks_from_dataset, 
    create_mobilenet_engine, 
    create_efficientnet_engine, 
    evaluate_model,
    get_callbacks,
    CLASS_NAMES
)

DATASET_DIR = r"E:\Project\dataset\bisindo\images\train"
# DATASET_DIR = r"E:\Project\comvis\train_model\bisindo\images\train"

print("Mulai ekstrak fitur 2 tangan dari gambar...")
X, y = extract_landmarks_from_dataset(DATASET_DIR, 'saved_models/landmarks_train.npz')

# Bagi data training & validasi
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = tf.keras.utils.to_categorical(y_train, len(CLASS_NAMES))
y_val_cat = tf.keras.utils.to_categorical(y_val, len(CLASS_NAMES))

# Simpan urutan abjad buat referensi web
np.save('saved_models/landmark_classifier_classes.npy', CLASS_NAMES)

# ============================================================
# 1. TRAIN MOBILENET V2 (Fast Engine)
# ============================================================
print("\n" + "="*60)
print("TRAINING MOBILENET V2 (REAL-TIME ENGINE)")
print("="*60)
model_fast = create_mobilenet_engine(len(CLASS_NAMES))
# Pake callbacks biar cuma nyimpen model pas akurasinya naik
callbacks_fast = get_callbacks('saved_models/mobilenet_landmark.keras', patience=15)

model_fast.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), 
               epochs=50, batch_size=32, callbacks=callbacks_fast)

# Panggil fungsi evaluasi buat bikin Confusion Matrix
evaluate_model(model_fast, X_val, y_val)

# ============================================================
# 2. TRAIN EFFICIENTNET B0 (Accurate Engine)
# ============================================================
print("\n" + "="*60)
print("TRAINING EFFICIENTNET B0 (ACCURATE ENGINE)")
print("="*60)
model_acc = create_efficientnet_engine(len(CLASS_NAMES))
callbacks_acc = get_callbacks('saved_models/efficientnet_landmark.keras', patience=15)

model_acc.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), 
              epochs=70, batch_size=32, callbacks=callbacks_acc)

# Panggil fungsi evaluasi buat bikin Confusion Matrix
evaluate_model(model_acc, X_val, y_val)

print("\nSemua beres! dan siap di-deploy ke web! 🚀")