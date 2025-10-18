# train_rnn_variant.py
import argparse
import importlib
import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ===== 1) Импорт генератора ряда из выбранного varX.py =====
def load_variant_module(variant: str):
    """
    variant: строка вида 'var1', 'var2', ..., 'var8' (без .py)
    В модуле ожидается функция gen_sequence(seq_len=...).
    """
    try:
        mod = importlib.import_module(variant)
    except ModuleNotFoundError as e:
        raise SystemExit(f"Не найден модуль {variant}. Убедись, что {variant}.py лежит рядом. Ошибка: {e}")
    if not hasattr(mod, "gen_sequence"):
        raise SystemExit(f"В модуле {variant} нет функции gen_sequence(seq_len=...). Добавь её.")
    return mod

# ===== 2) Окна и разбиение =====
def make_windows(series: np.ndarray, window_size: int, horizon: int = 1, stride: int = 1):
    """
    Превращает одномерный ряд в (X, y), где X.shape = (n, window, 1), y.shape = (n, horizon)
    """
    X, y = [], []
    end = len(series) - window_size - horizon + 1
    for start in range(0, max(end, 0), stride):
        x = series[start:start + window_size]
        t = series[start + window_size:start + window_size + horizon]
        X.append(x)
        y.append(t)
    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.float32)
    return X, y

@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_end_idx: int
    val_end_idx: int

def chronological_split(X, y, train_ratio=0.7, val_ratio=0.15) -> Split:
    n = X.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return Split(
        X_train=X[:train_end], y_train=y[:train_end],
        X_val=X[train_end:val_end], y_val=y[train_end:val_end],
        X_test=X[val_end:], y_test=y[val_end:],
        train_end_idx=train_end, val_end_idx=val_end
    )

# ===== 3) Модели =====
def build_model(window_size: int, horizon: int, kind: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = inputs
    if kind == "gru":
        x = tf.keras.layers.GRU(64, return_sequences=True)(x)
        x = tf.keras.layers.GRU(64)(x)
    elif kind == "lstm":
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(64)(x)
    elif kind == "gru_lstm":
        x = tf.keras.layers.GRU(64, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(64)(x)
    else:
        raise ValueError("model kind must be one of: gru | lstm | gru_lstm")

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

# ===== 4) Плоты =====
def plot_learning_curves(history, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Learning curves (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()

def plot_test_predictions(series, window_size, split: Split, preds_test, save_path=None):
    # индексы целей теста в исходном ряду
    start_target_idx = window_size + split.val_end_idx
    idx = np.arange(start_target_idx, start_target_idx + len(split.y_test))

    plt.figure(figsize=(12, 5))
    plt.plot(series, label="Исходная последовательность")
    plt.scatter(idx, split.y_test.squeeze(), s=12, label="Истина (test)")
    plt.scatter(idx, preds_test.squeeze(), s=12, label="Предсказание (test)")
    plt.title("Прогноз на тестовой выборке")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()

def plot_autoregressive(series, series_norm, window, split: Split, model, sigma, mu, save_path=None):
    steps = len(split.y_test)
    context = series_norm[split.val_end_idx:split.val_end_idx + window].copy()
    auto_preds = []
    for _ in range(steps):
        x = context[-window:].reshape(1, window, 1)
        yhat = model.predict(x, verbose=0)[0, 0]
        auto_preds.append(yhat)
        context = np.append(context, yhat)
    auto_denorm = np.array(auto_preds) * sigma + mu

    start_target_idx = window + split.val_end_idx
    idx = np.arange(start_target_idx, start_target_idx + steps)

    plt.figure(figsize=(12, 5))
    plt.plot(series, label="Исходная последовательность")
    plt.scatter(idx, series[idx], s=12, label="Истина (test)")
    plt.scatter(idx, auto_denorm, s=12, label="Авторегрессия (test)")
    plt.title("Авторегрессионный прогноз на тестовой части")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()

# ===== 5) Main =====
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="Имя модуля varX без .py (например, var1, var2, ...)")
    p.add_argument("--seq_len", type=int, default=2000)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["gru", "lstm", "gru_lstm"], default="gru_lstm")
    p.add_argument("--out_dir", default=".")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # сиды
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # импорт варианта
    mod = load_variant_module(args.variant)

    # генерация
    series = mod.gen_sequence(args.seq_len).astype(np.float32)

    # нормализация
    mu, sigma = series.mean(), series.std() + 1e-8
    series_norm = (series - mu) / sigma

    # окна
    X, y = make_windows(series_norm, window_size=args.window, horizon=args.horizon, stride=args.stride)

    # split
    split = chronological_split(X, y, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # модель
    model = build_model(args.window, args.horizon, args.model)
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, f"{args.variant}_{args.model}.keras"),
            monitor="val_loss", save_best_only=True
        ),
    ]

    # обучение
    history = model.fit(
        split.X_train, split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False
    )

    # оценка
    test_mse, test_mae = model.evaluate(split.X_test, split.y_test, verbose=0)
    print(f"[{args.variant}] model={args.model}  TEST: MSE={test_mse:.6f}  MAE={test_mae:.6f}")

    # предсказания
    preds_test = model.predict(split.X_test, verbose=0)

    # пути сохранения с пометкой _varX
    out_dir = os.path.join(args.out_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    variant_tag = f"_{args.variant}"

    plot_learning_curves(history, save_path=os.path.join(out_dir, f"learning_curves{variant_tag}.png"))
    plot_test_predictions(
        series, args.window, split, preds_test,
        save_path=os.path.join(out_dir, f"forecast_test{variant_tag}.png")
    )
    plot_autoregressive(
        series, series_norm, args.window, split, model, sigma, mu,
        save_path=os.path.join(out_dir, f"forecast_test_autoregressive{variant_tag}.png")
    )

    model.save(os.path.join(out_dir, f"model{variant_tag}.keras"))

    print("Сохранено в results/:")
    print(f"  - learning_curves{variant_tag}.png")
    print(f"  - forecast_test{variant_tag}.png")
    print(f"  - forecast_test_autoregressive{variant_tag}.png")
    print(f"  - model{variant_tag}.keras")


if __name__ == "__main__":
    main()
