# Примечания к задачам:
# Необходимо использовать модуль numpy
# Все данные должны считываться из файла в виде массива numpy
# Результаты необходимо сохранять в файл
# Задача 1
# Дано множество из p матриц (n,n) и множество из p векторов (n,1). Написать функцию для рассчета суммы p произведений матриц (результат имеет размерность (n,1))


import numpy as np

def calc_sum_products(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Вычисляет сумму произведений p матриц (n×n) и p векторов (n×1)."""
    return np.sum(np.matmul(matrices, vectors), axis=0)


def read_data(filename: str):
    """Считывает данные из файла input.txt."""
    with open(filename, 'r') as f:
        lines = f.read().strip().split()
        p, n = map(int, lines[:2])
        data = np.array(lines[2:], dtype=int)
        total_matrix_vals = p * n * n
        total_vector_vals = p * n
        matrices = data[:total_matrix_vals].reshape(p, n, n)
        vectors = data[total_matrix_vals:total_matrix_vals + total_vector_vals].reshape(p, n, 1)
    return matrices, vectors


def save_result(filename: str, result: np.ndarray):
    """Сохраняет результат в файл."""
    np.savetxt(filename, result, fmt='%d')


def generate_input_file(filename: str):
    """Генерирует пример данных (из условия задачи)."""
    with open(filename, "w") as f:
        f.write("3 4\n")  # p=3, n=4

        # матрицы
        for m in range(3):
            for i in range(4):
                row = [str(m * 16 + i * 4 + j) for j in range(4)]
                f.write(" ".join(row) + "\n")

        # векторы
        values = list(range(12))
        for v in values:
            f.write(f"{v}\n")


# Дано:
p = 3
n = 4

a = np.array([
    [[0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11],
     [12, 13, 14, 15]],

    [[16, 17, 18, 19],
     [20, 21, 22, 23],
     [24, 25, 26, 27],
     [28, 29, 30, 31]],

    [[32, 33, 34, 35],
     [36, 37, 38, 39],
     [40, 41, 42, 43],
     [44, 45, 46, 47]]
])

b = np.array([
    [[0],
     [1],
     [2],
     [3]],

    [[4],
     [5],
     [6],
     [7]],

    [[8],
     [9],
     [10],
     [11]]
])

# === Расчёт ===
result = np.sum(np.matmul(a, b), axis=0)

# === Сохранение результата в нужном формате ===
with open("1_task_result.txt", "w") as f:
    f.write("результат = \n")
    f.write(np.array2string(result, separator=' '))

print('--------------------------------------')
print('1 task')
print("✅ Результат сохранён в файл 1_task_result.txt")


# Задача 2
# Написать функцию преобразовывающую вектор чисел в матрицу бинарных представлений.
def vector_to_binary_matrix(vec: np.ndarray) -> np.ndarray:
    """
    Преобразует вектор чисел в матрицу их бинарных представлений.
    Например: [1, 2, 3] -> [[0,0,1], [0,1,0], [0,1,1]]
    """
    # Находим, сколько бит нужно для самого большого числа
    max_bits = int(np.ceil(np.log2(np.max(vec) + 1))) if np.max(vec) > 0 else 1

    # Формируем бинарную матрицу
    binary_matrix = np.array(
        [[int(b) for b in np.binary_repr(x, width=max_bits)] for x in vec]
    )

    return binary_matrix


print('--------------------------------------')
print('2 task')

# === Пример ===
vector = np.array([1, 2, 3, 4, 7, 8])
binary_matrix = vector_to_binary_matrix(vector)

print("Входной вектор:")
print(vector)
print("\nБинарная матрица:")
print(binary_matrix)

# === Сохранение результата ===
with open("2_task_result.txt", "w") as f:
    f.write("результат = \n")
    f.write(np.array2string(binary_matrix, separator=' '))

print("✅ Результат сохранён в файл 2_task_result.txt")
# Задача 3
# Написать функцию, которая возвращает все уникальные строки матрицы


def unique_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Возвращает все уникальные строки матрицы.
    """
    return np.unique(matrix, axis=0)


# === Пример ===
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3],
    [7, 8, 9],
    [4, 5, 6]
])

print('--------------------------------------')
print('3 task')

unique_matrix = unique_rows(matrix)

print("Исходная матрица:")
print(matrix)
print("\nУникальные строки:")
print(unique_matrix)

# === Сохранение результата ===
with open("3_task_result.txt", "w") as f:
    f.write("результат = \n")
    f.write(np.array2string(unique_matrix, separator=' '))

print("✅ Результат сохранён в файл 3_task_result.txt")
# Задача 4
# Написать функцию, которая заполняет матрицу с размерами (M,N) случайными числами распределенными по нормальному закону. Затем считает мат. ожидание и дисперсию для каждого из столбцов, а также строит для каждой строки стоит гистограмму значений (использовать функцию hist из модуля matplotlib.plot)
import matplotlib.pyplot as plt

def gen_normal_matrix(M: int, N: int, mean: float = 0.0, std: float = 1.0, seed: int | None = 42) -> np.ndarray:
    """
    Генерирует матрицу (M, N) из нормального распределения N(mean, std^2).
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(loc=mean, scale=std, size=(M, N))

def column_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (mean, var) по столбцам: shape (N,), (N,)
    Дисперсия — популяционная (ddof=0).
    """
    means = X.mean(axis=0)
    vars_ = X.var(axis=0, ddof=0)
    return means, vars_

def save_stats(filename: str, means: np.ndarray, vars_: np.ndarray) -> None:
    """
    Сохраняет результаты в файл в читаемом формате.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("результат = \n")
        f.write("Мат. ожидание по столбцам:\n")
        f.write(np.array2string(means, precision=6, separator=' ') + "\n\n")
        f.write("Дисперсия по столбцам:\n")
        f.write(np.array2string(vars_, precision=6, separator=' ') + "\n")

def save_row_histograms(X: np.ndarray, bins: int = 20, prefix: str = "4_task_hist_row_") -> None:
    """
    Строит и сохраняет гистограммы для каждой строки матрицы X.
    Файлы: 4_task_hist_row_0.png, 4_task_hist_row_1.png, ...
    """
    M = X.shape[0]
    for i in range(M):
        plt.figure()
        plt.hist(X[i, :], bins=bins)
        plt.title(f"Row {i} histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"{prefix}{i}.png", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    print('--------------------------------------')
    print('4 task')

    # Параметры матрицы и распределения — при необходимости поменяй
    M, N = 6, 50          # матрица 6x50
    MU, SIGMA = 0.0, 1.0  # N(0, 1)
    SEED = 42

    # 1) Генерация данных
    X = gen_normal_matrix(M, N, mean=MU, std=SIGMA, seed=SEED)

    # 2) Статистики по столбцам
    means, vars_ = column_stats(X)

    # 3) Сохранение результатов (текстовый файл)
    save_stats("../4_task_result.txt", means, vars_)

    # 4) Сохранение гистограмм по строкам (PNG-файлы)
    save_row_histograms(X, bins=20, prefix="4_task_hist_row_")

    print("✅ Результаты сохранены в '4_task_result.txt' и гистограммы '4_task_hist_row_*.png'")
# Задача 5
# Написать функцию, которая заполняет матрицу (M,N) в шахматном порядке заданными числами a и b.
import numpy as np

print('--------------------------------------')
print('5 task')

def chess_matrix(M: int, N: int, a: int, b: int) -> np.ndarray:
    """
    Создает матрицу MxN, заполненную числами a и b в шахматном порядке.
    """
    matrix = np.zeros((M, N), dtype=int)
    for i in range(M):
        for j in range(N):
            matrix[i, j] = a if (i + j) % 2 == 0 else b
    return matrix


# === Пример ===
M, N = 6, 8
a, b = 1, 0

result = chess_matrix(M, N, a, b)

print("✅ Матрица с шахматным заполнением создана.")

# === Сохранение результата ===
with open("5_task_result.txt", "w", encoding="utf-8") as f:
    f.write("результат = \n")
    f.write(np.array2string(result, separator=' '))

print("✅ Результат сохранён в файл 5_task_result.txt")

# Задача 6
# Написать функцию, которая возвращает тензор представляющий изображение круга с заданным цветом и радиусом в схеме rgd на черном фоне.
import numpy as np
import matplotlib.pyplot as plt

print('--------------------------------------')
print('6 task')

def draw_circle_tensor(size: int, radius: int, color: tuple[float, float, float]) -> np.ndarray:
    """
    Возвращает тензор (size, size, 3), представляющий изображение круга
    с заданным радиусом и цветом (RGB) на черном фоне.

    color — кортеж (R, G, B) значений от 0 до 1.
    """
    img = np.zeros((size, size, 3), dtype=float)
    center = (size // 2, size // 2)

    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

    for c in range(3):
        img[..., c][mask] = color[c]

    return img


# === Пример ===
size = 256
radius = 80
color = (1.0, 0.0, 0.0)  # красный

circle_img = draw_circle_tensor(size, radius, color)

# === Визуализация ===
plt.imshow(circle_img)
plt.axis("off")
plt.title("Circle RGB Tensor")
plt.show()

# === Сохранение результата ===
# Для текстового формата можно записать только часть тензора (иначе файл будет огромным)
with open("6_task_result.txt", "w", encoding="utf-8") as f:
    f.write("результат = тензор изображения круга (RGB)\n")
    f.write(f"Размер: {size}x{size}, радиус: {radius}, цвет: {color}\n")
    f.write("Пример значений центрального фрагмента:\n")
    center_crop = circle_img[size//2-3:size//2+4, size//2-3:size//2+4, :]
    f.write(np.array2string(center_crop, precision=2, separator=' '))

print("✅ Результат сохранён в файл 6_task_result.txt")

# Задача 7
# Написать функцию, которая стандартизирует все значения тензор (отнять мат. ожидание и поделить на СКО)
import numpy as np

print('--------------------------------------')
print('7 task')

def standardize_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Стандартизирует значения тензора:
    (x - mean) / std
    где mean — мат. ожидание, std — стандартное отклонение.
    """
    mean = np.mean(tensor)
    std = np.std(tensor)
    standardized = (tensor - mean) / std
    return standardized, mean, std


# === Пример ===
# создадим тестовый тензор
tensor = np.random.randint(0, 255, size=(3, 3, 3)).astype(float)

standardized_tensor, mean, std = standardize_tensor(tensor)

print(f"✅ Среднее значение: {mean:.4f}")
print(f"✅ Стандартное отклонение: {std:.4f}")
print("✅ Тензор стандартизирован.")

# === Сохранение результата ===
with open("7_task_result.txt", "w", encoding="utf-8") as f:
    f.write("результат = стандартизированный тензор\n")
    f.write(f"Мат. ожидание (mean): {mean:.6f}\n")
    f.write(f"СКО (std): {std:.6f}\n")
    f.write("Стандартизированные значения (фрагмент):\n")
    f.write(np.array2string(standardized_tensor, precision=3, separator=' '))

print("✅ Результат сохранён в файл 7_task_result.txt")

# Задача 8
# Написать функцию, выделяющую часть матрицы фиксированного размера с центром в данном элементе (дополненное значением fill если необходимо)
import numpy as np

print('--------------------------------------')
print('8 task')

def extract_patch(matrix: np.ndarray, center: tuple[int, int], patch_size: tuple[int, int], fill: float = 0) -> np.ndarray:
    """
    Выделяет часть матрицы (patch) фиксированного размера patch_size=(h, w)
    с центром в элементе center=(row, col).
    Если окно выходит за границы матрицы — дополняет значением fill.
    """
    M, N = matrix.shape
    h, w = patch_size
    cy, cx = center

    patch = np.full((h, w), fill, dtype=matrix.dtype)

    # границы исходной матрицы
    y1 = cy - h // 2
    y2 = cy + h // 2 + (h % 2)
    x1 = cx - w // 2
    x2 = cx + w // 2 + (w % 2)

    # пересечение с границами матрицы
    src_y1 = max(0, y1)
    src_y2 = min(M, y2)
    src_x1 = max(0, x1)
    src_x2 = min(N, x2)

    # куда вставлять в patch
    dst_y1 = src_y1 - y1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = src_x1 - x1
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    patch[dst_y1:dst_y2, dst_x1:dst_x2] = matrix[src_y1:src_y2, src_x1:src_x2]
    return patch


# === Пример ===
matrix = np.arange(1, 26).reshape(5, 5)
center = (0, 0)         # верхний левый угол
patch_size = (3, 3)
fill_value = 0

patch = extract_patch(matrix, center, patch_size, fill_value)

print("✅ Исходная матрица:")
print(matrix)
print("\n✅ Извлечённый фрагмент:")
print(patch)

# === Сохранение результата ===
with open("8_task_result.txt", "w", encoding="utf-8") as f:
    f.write("результат = извлечённый фрагмент матрицы\n")
    f.write(f"Центр: {center}, Размер: {patch_size}, fill={fill_value}\n")
    f.write(np.array2string(patch, separator=' '))

print("✅ Результат сохранён в файл 8_task_result.txt")

# Задача 9
# Написать функцию, которая находит самое часто встречающееся число в каждой строке матрицы и возвращает массив этих значений
import numpy as np
from collections import Counter

print('--------------------------------------')
print('9 task')

def most_frequent_per_row(matrix: np.ndarray) -> np.ndarray:
    """
    Для каждой строки матрицы находит самое часто встречающееся число.
    Если несколько значений встречаются одинаково часто — берётся первое по порядку.
    Возвращает вектор (n_rows, ).
    """
    result = []
    for row in matrix:
        counts = Counter(row)
        most_common = counts.most_common(1)[0][0]
        result.append(most_common)
    return np.array(result)


# === Пример ===
matrix = np.array([
    [1, 2, 2, 3, 3, 3],
    [4, 4, 5, 5, 4, 6],
    [7, 8, 7, 8, 7, 9],
    [1, 1, 2, 2, 3, 3]
])

result = most_frequent_per_row(matrix)

print("✅ Исходная матрица:")
print(matrix)
print("\n✅ Наиболее часто встречающиеся числа по строкам:")
print(result)

# === Сохранение результата ===
with open("9_task_result.txt", "w", encoding="utf-8") as f:
    f.write("результат = наиболее часто встречающиеся числа по строкам\n")
    f.write(np.array2string(result, separator=' '))

print("✅ Результат сохранён в файл 9_task_result.txt")

# Задача 10
# Дан трёхмерный массив, содержащий изображение, размера (height, width, numChannels), а также вектор длины numChannels. Написать функцию, которая складывает каналы изображения с указанными весами, и возвращает результат в
# виде матрицы размера (height, width)
import numpy as np

print('--------------------------------------')
print('10 task')

def weighted_channel_sum(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Складывает каналы 3D-изображения с заданными весами.

    image — numpy массив формы (height, width, numChannels)
    weights — вектор длины numChannels

    Возвращает 2D матрицу (height, width)
    """
    if image.shape[2] != len(weights):
        raise ValueError("Длина вектора весов должна совпадать с числом каналов изображения")

    # np.tensordot удобно для взвешенной суммы по последней оси
    result = np.tensordot(image, weights, axes=([2], [0]))
    return result


# === Пример ===
height, width, numChannels = 4, 4, 3
image = np.random.randint(0, 256, size=(height, width, numChannels)).astype(float)

# например, RGB-веса (приближённое преобразование в оттенки серого)
weights = np.array([0.2989, 0.5870, 0.1140])

result = weighted_channel_sum(image, weights)

print("✅ Исходное изображение (фрагмент):")
print(image[:2, :2, :])
print("\n✅ Весовой вектор:")
print(weights)
print("\n✅ Результирующая матрица (взвешенная сумма каналов):")
print(result[:2, :2])

# === Сохранение результата ===
with open("10_task_result.txt", "w", encoding="utf-8") as f:
    f.write("результат = матрица (height, width) после взвешенной суммы каналов\n")
    f.write(f"Размер входного изображения: {image.shape}\n")
    f.write(f"Веса каналов: {weights.tolist()}\n\n")
    f.write("Фрагмент результата:\n")
    f.write(np.array2string(result[:5, :5], precision=2, separator=' '))

print("✅ Результат сохранён в файл 10_task_result.txt")

