# File: train_baru.py
import numpy as np
from sklearn.model_selection import train_test_split
# Kita import juga fungsi evaluate_model dan plot_training_history nya
from asl_modules import extract_landmarks_from_dataset, train_landmark_model, evaluate_model, plot_training_history

# Path Dataset lu (Pastiin udah bener ya ngab)
DATASET_DIR = r"E:\Project\dataset\bisindo\images\train"

print("Mulai ekstrak fitur 2 tangan dari gambar...")
X, y = extract_landmarks_from_dataset(DATASET_DIR, 'saved_models/landmarks_train.npz')

print("Mulai training model baru...")
model, history = train_landmark_model(X, y, epochs=50, save_path='saved_models/landmark_classifier.keras')

# ============================================================
# FITUR EVALUASI (Accuracy, Precision, Recall, Confusion Matrix)
# ============================================================
print("\n" + "="*60)
print("MENGHITUNG METRIK EVALUASI")
print("="*60)

# Kita mecah data lagi buat ngetes dengan persentase (20%) 
# dan random state yang PERSIS SAMA kayak pas training di modul lu
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Panggil fungsi evaluasi (munculin Precision, Recall, Accuracy, & Confusion Matrix)
evaluate_model(model, X_val, y_val)

# 2. Gambar grafik pergerakan Loss & Accuracy pas proses training
plot_training_history(history)

print("\nBERES NGAB! Cek folder comvis lu, ada file 'confusion_matrix.png' dan 'training_history.png' tuh!")