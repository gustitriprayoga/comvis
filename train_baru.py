"""
BISINDO Recognition - Training Script
Transfer Learning 2 Fase: MobileNetV3 vs DenseNet121

Fase 1: Freeze base model, train head only (cepat, ~3-5 epoch)
Fase 2: Unfreeze beberapa layer, fine-tune (10-15 epoch)
"""
import os
import time
import numpy as np
import tensorflow as tf
from asl_modules import (
    create_data_generators,
    create_mobilenetv3_model,
    create_densenet121_model,
    unfreeze_model,
    evaluate_model,
    get_callbacks,
    plot_training_history,
    CLASS_NAMES
)

# ============================================================
# KONFIGURASI
# ============================================================
DATASET_DIR = r"E:\Project\dataset\bisindo\images\train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Fase 1: Train head only (base frozen) — CEPAT
EPOCHS_FASE1 = 5

# Fase 2: Fine-tune beberapa layer — AKURASI NAIK
EPOCHS_FASE2_MOBILENET = 15
EPOCHS_FASE2_DENSENET = 20

# Jumlah layer terakhir base model yang di-unfreeze saat fine-tune
UNFREEZE_LAYERS_MOBILENET = 20
UNFREEZE_LAYERS_DENSENET = 30

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "="*60)
print("  LOADING DATASET")
print("="*60)

train_gen, val_gen, test_gen, class_indices = create_data_generators(
    DATASET_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    test_split=0.1
)

num_classes = len(class_indices)
print(f"\nJumlah kelas: {num_classes}")
print(f"Kelas: {list(class_indices.keys())}")

# Simpan urutan abjad buat referensi web
os.makedirs('saved_models', exist_ok=True)
np.save('saved_models/landmark_classifier_classes.npy', list(class_indices.keys()))

total_start = time.time()

# ============================================================
# 2. TRAIN MOBILENETV3 (2 Fase)
# ============================================================
print("\n" + "="*60)
print("  TRAINING MOBILENETV3")
print("="*60)

model_mobilenet = create_mobilenetv3_model(num_classes, IMG_SIZE)

# --- FASE 1: Train head only (base frozen) ---
print("\n--- FASE 1: Training Head (base frozen) ---")
callbacks_f1 = get_callbacks('saved_models/mobilenetv3_model.keras', patience=5)

start_mn = time.time()
history_f1 = model_mobilenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FASE1,
    callbacks=callbacks_f1
)

# --- FASE 2: Fine-tune beberapa layer terakhir ---
print(f"\n--- FASE 2: Fine-tuning (unfreeze {UNFREEZE_LAYERS_MOBILENET} layer terakhir) ---")
model_mobilenet = unfreeze_model(model_mobilenet,
                                 num_layers_to_unfreeze=UNFREEZE_LAYERS_MOBILENET,
                                 learning_rate=1e-5)
callbacks_f2 = get_callbacks('saved_models/mobilenetv3_model.keras', patience=7)

history_f2 = model_mobilenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FASE2_MOBILENET,
    callbacks=callbacks_f2
)

time_mn = time.time() - start_mn
print(f"\n[MobileNetV3] Total waktu training: {time_mn/60:.1f} menit")

# Gabung history kedua fase untuk plot
combined_history_mn = {
    'accuracy': history_f1.history['accuracy'] + history_f2.history['accuracy'],
    'val_accuracy': history_f1.history['val_accuracy'] + history_f2.history['val_accuracy'],
    'loss': history_f1.history['loss'] + history_f2.history['loss'],
    'val_loss': history_f1.history['val_loss'] + history_f2.history['val_loss'],
}

class CombinedHistory:
    def __init__(self, h): self.history = h

plot_training_history(CombinedHistory(combined_history_mn), model_name='MobileNetV3')
print("\n[MobileNetV3] Evaluasi...")
results_mn = evaluate_model(model_mobilenet, test_gen)

# ============================================================
# 3. TRAIN DENSENET121 (2 Fase)
# ============================================================
print("\n" + "="*60)
print("  TRAINING DENSENET121")
print("="*60)

model_densenet = create_densenet121_model(num_classes, IMG_SIZE)

# --- FASE 1: Train head only (base frozen) ---
print("\n--- FASE 1: Training Head (base frozen) ---")
callbacks_f1 = get_callbacks('saved_models/densenet121_model.keras', patience=5)

start_dn = time.time()
history_f1 = model_densenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FASE1,
    callbacks=callbacks_f1
)

# --- FASE 2: Fine-tune beberapa layer terakhir ---
print(f"\n--- FASE 2: Fine-tuning (unfreeze {UNFREEZE_LAYERS_DENSENET} layer terakhir) ---")
model_densenet = unfreeze_model(model_densenet,
                                num_layers_to_unfreeze=UNFREEZE_LAYERS_DENSENET,
                                learning_rate=1e-5)
callbacks_f2 = get_callbacks('saved_models/densenet121_model.keras', patience=7)

history_f2 = model_densenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FASE2_DENSENET,
    callbacks=callbacks_f2
)

time_dn = time.time() - start_dn
print(f"\n[DenseNet121] Total waktu training: {time_dn/60:.1f} menit")

# Gabung history kedua fase
combined_history_dn = {
    'accuracy': history_f1.history['accuracy'] + history_f2.history['accuracy'],
    'val_accuracy': history_f1.history['val_accuracy'] + history_f2.history['val_accuracy'],
    'loss': history_f1.history['loss'] + history_f2.history['loss'],
    'val_loss': history_f1.history['val_loss'] + history_f2.history['val_loss'],
}

plot_training_history(CombinedHistory(combined_history_dn), model_name='DenseNet121')
print("\n[DenseNet121] Evaluasi...")
results_dn = evaluate_model(model_densenet, test_gen)

# ============================================================
# 4. RINGKASAN PERBANDINGAN
# ============================================================
total_time = time.time() - total_start

print("\n" + "="*60)
print("  RINGKASAN PERBANDINGAN")
print("="*60)
print(f"{'Metrik':<20} {'MobileNetV3':>15} {'DenseNet121':>15}")
print("-"*50)
print(f"{'Accuracy':<20} {results_mn['accuracy']:>14.4f} {results_dn['accuracy']:>14.4f}")
print(f"{'Precision':<20} {results_mn['precision']:>14.4f} {results_dn['precision']:>14.4f}")
print(f"{'Recall':<20} {results_mn['recall']:>14.4f} {results_dn['recall']:>14.4f}")
print(f"{'F1-Score':<20} {results_mn['f1_score']:>14.4f} {results_dn['f1_score']:>14.4f}")
print(f"{'Inference (det/img)':<20} {results_mn['inference_time']:>14.6f} {results_dn['inference_time']:>14.6f}")
print(f"{'Waktu Training':<20} {time_mn/60:>13.1f}m {time_dn/60:>13.1f}m")
print("-"*50)
print(f"{'Total Waktu':<20} {total_time/60:>14.1f} menit")

print(f"\nSemua file tersimpan di: saved_models/ dan saved_generate/")
print("Siap di-deploy ke web! 🚀")