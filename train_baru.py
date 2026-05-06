import numpy as np
import tensorflow as tf
from asl_modules import (
    create_data_generators,
    create_mobilenetv3_model,
    create_densenet121_model,
    evaluate_model,
    get_callbacks,
    plot_training_history,
    CLASS_NAMES
)

DATASET_DIR = r"E:\Project\dataset\bisindo\images\train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_MOBILENET = 50
EPOCHS_DENSENET = 70

# ============================================================
# 1. LOAD DATA
# ============================================================
train_gen, val_gen, test_gen, class_indices = create_data_generators(
    DATASET_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    test_split=0.1
)

# Simpan urutan abjad buat referensi web
np.save('saved_models/landmark_classifier_classes.npy', list(class_indices.keys()))

# ============================================================
# 2. TRAIN MOBILENETV3
# ============================================================
print("\n" + "="*60)
print("TRAINING MOBILENETV3")
print("="*60)
model_mobilenet = create_mobilenetv3_model(len(CLASS_NAMES), IMG_SIZE)
callbacks_mobilenet = get_callbacks('saved_models/mobilenetv3_model.keras', patience=15)

history_mobilenet = model_mobilenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_MOBILENET,
    callbacks=callbacks_mobilenet
)

plot_training_history(history_mobilenet)
evaluate_model(model_mobilenet, test_gen)

# ============================================================
# 3. TRAIN DENSENET121
# ============================================================
print("\n" + "="*60)
print("TRAINING DENSENET121")
print("="*60)
model_densenet = create_densenet121_model(len(class_indices), IMG_SIZE)
callbacks_densenet = get_callbacks('saved_models/densenet121_model.keras', patience=15)

history_densenet = model_densenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_DENSENET,
    callbacks=callbacks_densenet
)

plot_training_history(history_densenet)
evaluate_model(model_densenet, test_gen)

print("\nSemua beres! dan siap di-deploy ke web! 🚀")