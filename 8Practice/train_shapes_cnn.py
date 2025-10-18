# train_shapes_cnn.py
# Usage:
#   python train_shapes_cnn.py --variant 1 --out_prefix var1 --module_dir ../6Practice
#   python train_shapes_cnn.py --variant 2 --out_prefix var2 --module_dir ../6Practice
#   python train_shapes_cnn.py --variant 6 --out_prefix var6 --module_dir ../6Practice

import argparse
import os
import sys
import importlib.util
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import inspect

# --------------------
# Reproducibility
# --------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def preload_gens(module_dir: str) -> None:
    """
    Если varX.py делает `import gens`, предварительно загрузим gens.py
    и положим его в sys.modules['gens'], чтобы импорт сработал.
    """
    gens_path = os.path.join(module_dir, "gens.py")
    if not os.path.isfile(gens_path):
        return
    spec = importlib.util.spec_from_file_location("gens", gens_path)
    gens_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gens_mod)  # type: ignore[attr-defined]
    sys.modules["gens"] = gens_mod


def load_gen_data(module_path: str):
    """
    Динамически загружает varX.py и возвращает ссылку на функцию gen_data()
    (ожидается сигнатура: -> (X: (N,H,W) float32/bool, y: (N,1) строковые метки))
    """
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"var-file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("var_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "gen_data"):
        raise RuntimeError(f"{module_path} does not define gen_data(...)")

    print("gen_data loaded from:", inspect.getsourcefile(mod.gen_data))  # type: ignore[arg-type]
    return mod.gen_data


def build_cnn(input_shape, n_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="shapes_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, required=True, help="номер varX.py (1..7)")
    ap.add_argument("--out_prefix", type=str, required=True, help="префикс имён файлов: var1/var2/var6...")
    ap.add_argument("--module_dir", type=str, default=os.path.dirname(__file__),
                    help="папка, где лежат var*.py и gens.py (например, ../6Practice)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # где лежат var*.py
    module_dir_abs = os.path.abspath(args.module_dir)
    sys.path.insert(0, module_dir_abs)
    preload_gens(module_dir_abs)  # чтобы `import gens` внутри varX.py сработал

    # путь к var-файлу
    module_path = os.path.join(module_dir_abs, f"var{args.variant}.py")
    gen_data = load_gen_data(module_path)

    # === Сохраняем ВСЁ здесь (текущая папка, например 8Practice) ===
    save_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"[INFO] Outputs will be saved to: {save_dir}")

    # -------------------------
    # 1) Генерация
    # -------------------------
    X, y = gen_data()         # ожидается X: (N,H,W), y: (N,1) строковые метки
    y = y.reshape(-1)
    X = X.astype("float32")[..., np.newaxis]  # (N,H,W,1)

    # sanity
    uniq, cnt = np.unique(y, return_counts=True)
    print("Classes distribution:", dict(zip(uniq.tolist(), cnt.tolist())))
    print("X shape:", X.shape)

    # -------------------------
    # 2) Кодирование меток
    # -------------------------
    lbl = LabelEncoder()
    y_int = lbl.fit_transform(y)
    num_classes = len(lbl.classes_)
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes)
    print("Label classes:", lbl.classes_)

    # -------------------------
    # 3) Перемешивание и сплит
    # -------------------------
    idx = np.arange(len(y_int))
    np.random.shuffle(idx)
    X = X[idx]
    y_int = y_int[idx]
    y_cat = y_cat[idx]

    X_train, X_tmp, y_train, y_tmp, y_int_train, y_int_tmp = train_test_split(
        X, y_cat, y_int, test_size=0.30, random_state=SEED, stratify=y_int
    )
    X_val, X_test, y_val, y_test, y_int_val, y_int_test = train_test_split(
        X_tmp, y_tmp, y_int_tmp, test_size=0.50, random_state=SEED, stratify=y_int_tmp
    )
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # -------------------------
    # 4) Модель и обучение
    # -------------------------
    model = build_cnn(X.shape[1:], num_classes)
    model.summary()

    ckpt_path = os.path.join(save_dir, f"shapes_cnn_{args.out_prefix}.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # -------------------------
    # 5) Оценка и сохранения
    # -------------------------
    probs = model.predict(X_test, batch_size=64, verbose=0)
    pred = probs.argmax(axis=1)

    print("\nClassification report (test):")
    print(classification_report(y_int_test, pred, target_names=lbl.classes_, digits=4))

    cm = confusion_matrix(y_int_test, pred)
    print("Confusion matrix:\n", cm)

    # CSV с предсказаниями
    df = pd.DataFrame({
        "y_true": [lbl.classes_[i] for i in y_int_test],
        "y_pred": [lbl.classes_[i] for i in pred],
        **{f"p_{lbl.classes_[k]}": probs[:, k] for k in range(num_classes)},
    })
    csv_path = os.path.join(save_dir, f"shapes_test_predictions_{args.out_prefix}.csv")
    df.to_csv(csv_path, index=False)

    # Превью картинок
    try:
        n_show = 6
        plt.figure(figsize=(10, 4))
        for i in range(n_show):
            ax = plt.subplot(2, n_show // 2, i + 1)
            ax.imshow(X_test[i].squeeze(), cmap="gray")
            ax.set_title(f"true={lbl.classes_[y_int_test[i]]}\npred={lbl.classes_[pred[i]]}")
            ax.axis("off")
        plt.tight_layout()
        png_path = os.path.join(save_dir, f"preview_samples_{args.out_prefix}.png")
        plt.savefig(png_path, dpi=150)
    except Exception as e:
        print("Preview skipped:", e)

    print("\nSaved files:")
    print("  -", ckpt_path)
    print("  -", csv_path)
    print("  -", os.path.join(save_dir, f"preview_samples_{args.out_prefix}.png"))


if __name__ == "__main__":
    main()
