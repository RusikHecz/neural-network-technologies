# -*- coding: utf-8 -*-
# CNN for grayscale geometric shapes (Variant 1: Rectangle vs Circle)
# Requires: tensorflow>=2.x, keras; your local files: var7.py, gens.py

import os
import random
import numpy as np
import pandas as pd

# 1) Data: import generator (returns X: (N, H, W), y: (N, 1) with string labels)
from var7 import gen_data  # <-- из твоего var7.py (uses gens.py inside)

# 2) ML stack
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Generate data
# -------------------------
SIZE = 500          # сколько картинок сгенерировать (поровну классов)
IMG_SIZE = 50       # размер изображения (H=W=50)

X, y = gen_data(size=SIZE, img_size=IMG_SIZE)  # X: (N, H, W), y: (N, 1) strings
y = y.reshape(-1)                               # (N,)
print(f"Raw shapes: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

# CNN expects channel dimension; data are 0/1 grayscale
X = X.astype("float32")[..., np.newaxis]        # (N, H, W, 1)

# Encode string labels -> integers -> one-hot
lbl = LabelEncoder()
y_int = lbl.fit_transform(y)                    # e.g., {'Circle':0, 'Square':1}
num_classes = len(lbl.classes_)
y_cat = tf.keras.utils.to_categorical(y_int, num_classes)

# Important: data were not shuffled and classes were consecutive; shuffle before split
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
y_cat = y_cat[idx]
y_int = y_int[idx]
y_raw = y[idx]  # string labels aligned for later reporting

# Train/val/test split: 70/15/15
X_train, X_tmp, y_train, y_tmp, y_int_train, y_int_tmp, y_raw_train, y_raw_tmp = train_test_split(
    X, y_cat, y_int, y_raw, test_size=0.30, random_state=SEED, stratify=y_int
)
X_val, X_test, y_val, y_test, y_int_val, y_int_test, y_raw_val, y_raw_test = train_test_split(
    X_tmp, y_tmp, y_int_tmp, y_raw_tmp, test_size=0.50, random_state=SEED, stratify=y_int_tmp
)

print(f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

# -------------------------
# Build CNN
# -------------------------
def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=2):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3,3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(32, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="shapes_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=num_classes)
model.summary()

# -------------------------
# Train
# -------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("shapes_cnn_var7.h5", monitor="val_accuracy", save_best_only=True),
]

H = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)

# -------------------------
# Evaluate & report
# -------------------------
test_probs = model.predict(X_test, batch_size=64, verbose=0)
test_pred = test_probs.argmax(axis=1)
print("\nClassification report (test):")
print(classification_report(y_int_test, test_pred, target_names=lbl.classes_, digits=4))

cm = confusion_matrix(y_int_test, test_pred)
print("Confusion matrix:\n", cm)

# -------------------------
# Save predictions CSV
# -------------------------
pred_df = pd.DataFrame({
    "y_true": [lbl.classes_[i] for i in y_int_test],
    "y_pred": [lbl.classes_[i] for i in test_pred],
    "p_" + lbl.classes_[0]: test_probs[:, 0],
    "p_" + lbl.classes_[1]: test_probs[:, 1],
})
pred_df.to_csv("shapes_test_predictions_var7.csv", index=False)
print("\nSaved:")
print("  - shapes_cnn_var7.h5")
print("  - shapes_test_predictions_var7.csv")

# -------------------------
# (Optional) quick sanity check: a few samples
# -------------------------
try:
    import matplotlib.pyplot as plt
    n_show = 6
    fig, axes = plt.subplots(2, n_show//2, figsize=(10,4))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X_test[i].squeeze(), cmap="gray")
        ax.set_title(f"true={lbl.classes_[y_int_test[i]]}\npred={lbl.classes_[test_pred[i]]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("preview_samples_var7.png", dpi=150)
    print("  - preview_samples.png")
except Exception as e:
    print("Preview skipped:", e)
