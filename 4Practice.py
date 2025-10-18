# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------
# 1) Датасет и логические варианты
# ----------------------------

VARIANT = 1  # Поменяйте число 1..8, чтобы выбрать формулу

def labels_for_variant(X, variant: int) -> np.ndarray:
    """
    X: shape (N, 3), каждый столбец — {0,1} для (a,b,c).
    Возвращает y shape (N,1) в {0,1} согласно выбранному варианту.
    """
    a = X[:, 0].astype(bool)
    b = X[:, 1].astype(bool)
    c = X[:, 2].astype(bool)

    if variant == 1:
        # (a and b) or (a and c)
        y = (a & b) | (a & c)
    elif variant == 2:
        # (a or b) xor not(b and c)
        y = (a | b) ^ (~(b & c))
    elif variant == 3:
        # (a and b) or c
        y = (a & b) | c
    elif variant == 4:
        # (a or b) and (b or c)
        y = (a | b) & (b | c)
    elif variant == 5:
        # (a xor b) and (b xor c)
        y = (a ^ b) & (b ^ c)
    elif variant == 6:
        # (a and not b) or (c xor b)
        y = (a & (~b)) | (c ^ b)
    elif variant == 7:
        # (a or b) and (a xor not b)
        y = (a | b) & (a ^ (~b))
    elif variant == 8:
        # (a and c and b) xor (a or not b)
        y = (a & c & b) ^ (a | (~b))
    else:
        raise ValueError("variant must be 1..8")

    return y.astype(np.float32).reshape(-1, 1)


def make_dataset():
    """
    Все 8 троек (a,b,c) из {0,1}³, X shape (8,3), y shape (8,1)
    """
    X = np.array([[a, b, c] for a in (0,1) for b in (0,1) for c in (0,1)], dtype=np.float32)
    y = labels_for_variant(X, VARIANT)
    return X, y


# ----------------------------
# 2) Модель Keras (MLP)
# ----------------------------

def build_model(hidden_units: int = 6, hidden_activation: str = "tanh"):
    model = models.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(hidden_units, activation=hidden_activation),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.05),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ----------------------------
# 3) Активации (для симуляторов)
# ----------------------------

def act_forward(x: np.ndarray, activation: str) -> np.ndarray:
    if activation is None or activation == "linear":
        return x
    if activation == "relu":
        return np.maximum(0.0, x)
    if activation == "tanh":
        return np.tanh(x)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unsupported activation: {activation}")


# ----------------------------
# 4) Извлечение структуры Dense-слоёв и весов
# ----------------------------

def extract_dense_layers(model: tf.keras.Model):
    """
    Возвращает список описаний слоёв:
    [
      {
        "W": ndarray shape (in, out),
        "b": ndarray shape (out,),
        "activation": str  # linear/relu/tanh/sigmoid
      },
      ...
    ]
    Только для Dense-слоёв; остальные игнорируются.
    """
    layers_info = []
    for lyr in model.layers:
        if isinstance(lyr, layers.Dense):
            W, b = lyr.get_weights()  # W shape (in, out), b shape (out,)
            # Имя активации:
            act = getattr(lyr.activation, "__name__", None)
            layers_info.append({"W": W, "b": b, "activation": act})
    return layers_info


# ----------------------------
# 5) Симулятор №1: поэлементные операции (+ суммирование)
# ----------------------------

def forward_elementwise(X: np.ndarray, layers_info) -> np.ndarray:
    """
    Прямой проход без матричного умножения:
    для y = X @ W + b считаем столбец за столбцом как sum(X * W_col).
    """
    out = X.copy()
    for info in layers_info:
        W, b, act = info["W"], info["b"], info["activation"]
        # Собираем линейную комбинацию вручную по столбцам
        lin = np.zeros((out.shape[0], W.shape[1]), dtype=np.float32)
        for j in range(W.shape[1]):
            # поэлементное умножение + сумма по признакам
            # (out * W[:, j]) имеет shape (N, in); суммируем по оси признаков
            lin[:, j] = (out * W[:, j]).sum(axis=1) + b[j]
        out = act_forward(lin, act)
    return out


# ----------------------------
# 6) Симулятор №2: NumPy-операции над тензорами (матр. умножение)
# ----------------------------

def forward_numpy(X: np.ndarray, layers_info) -> np.ndarray:
    """
    Прямой проход c матричными операциями NumPy.
    """
    out = X.copy()
    for info in layers_info:
        W, b, act = info["W"], info["b"], info["activation"]
        lin = out @ W + b  # матричное умножение + broadcast
        out = act_forward(lin, act)
    return out


# ----------------------------
# 7) Сравнение до и после обучения
# ----------------------------

def compare(name: str, a: np.ndarray, b: np.ndarray):
    """
    Печатает метрики сравнения двух массивов одинаковой формы
    """
    diff = np.abs(a - b)
    print(f"{name}: max|diff|={diff.max():.6f}, mean|diff|={diff.mean():.6f}")


def main():
    # Подготовим данные
    X, y = make_dataset()
    print(f"Вариант {VARIANT}: логическая функция построена. Размеры: X={X.shape}, y={y.shape}\n")

    # Строим и компилируем модель
    model = build_model(hidden_units=6, hidden_activation="tanh")
    print("Слои модели:")
    for i, lyr in enumerate(model.layers):
        print(f"  {i}: {lyr.name} ({lyr.__class__.__name__}), activation={getattr(getattr(lyr,'activation',None),'__name__',None)}")
    print()

    # Получаем веса не обученной модели (инициализация)
    layers_info = extract_dense_layers(model)
    print("Проверка: формы весов по слоям (до обучения):")
    for i, info in enumerate(layers_info):
        print(f"  Dense#{i}: W{info['W'].shape}, b{info['b'].shape}, act={info['activation']}")
    print()

    # Прямой проход НЕобученной модели
    raw_pred_untrained = model.predict(X, verbose=0)

    # Эмуляция НЕобученной модели двумя способами
    sim1_untrained = forward_elementwise(X, layers_info)
    sim2_untrained = forward_numpy(X, layers_info)

    print("Сравнение до обучения (по вероятностям сигмоиды):")
    compare("Model vs Sim1(elementwise)", raw_pred_untrained, sim1_untrained)
    compare("Model vs Sim2(numpy)     ", raw_pred_untrained, sim2_untrained)
    print()

    # Обучаем на всём датасете (полное множество наблюдений)
    H = model.fit(X, y, epochs=600, batch_size=8, verbose=0)

    # Получаем веса после обучения
    layers_info_tr = extract_dense_layers(model)
    print("Проверка: формы весов по слоям (после обучения):")
    for i, info in enumerate(layers_info_tr):
        print(f"  Dense#{i}: W{info['W'].shape}, b{info['b'].shape}, act={info['activation']}")
    print()

    # Прямой проход обученной модели
    raw_pred_trained = model.predict(X, verbose=0)
    bin_pred_trained = (raw_pred_trained >= 0.5).astype(int)

    # Эмуляция обученной модели
    sim1_trained = forward_elementwise(X, layers_info_tr)
    sim2_trained = forward_numpy(X, layers_info_tr)

    print("Сравнение после обучения (по вероятностям сигмоиды):")
    compare("Model vs Sim1(elementwise)", raw_pred_trained, sim1_trained)
    compare("Model vs Sim2(numpy)     ", raw_pred_trained, sim2_trained)

    # Сравним классификацию (порог 0.5)
    sim1_cls = (sim1_trained >= 0.5).astype(int)
    sim2_cls = (sim2_trained >= 0.5).astype(int)

    acc_model = (bin_pred_trained == y).mean()
    acc_sim1  = (sim1_cls == y).mean()
    acc_sim2  = (sim2_cls == y).mean()

    print("\nТочность на обученном наборе (ожидается 1.00 для линейно-разделимых/представимых функций):")
    print(f"  Model acc: {acc_model:.2f}")
    print(f"  Sim1  acc: {acc_sim1:.2f}")
    print(f"  Sim2  acc: {acc_sim2:.2f}")

    print("\nПримеры предсказаний (a b c | y_true | y_pred):")
    for i in range(X.shape[0]):
        print(f"{int(X[i,0])} {int(X[i,1])} {int(X[i,2])} | "
              f"{int(y[i,0])} | "
              f"{int(bin_pred_trained[i,0])}")

if __name__ == "__main__":
    # Для воспроизводимости (по желанию)
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
