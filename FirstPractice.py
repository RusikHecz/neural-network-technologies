# Задача №1
# Написать функцию на вход которой подается строка, состоящая из латинских букв. Функция должна вернуть количество гласных букв (a, e, i, o, u) в этой строке.
def count_vowels(s: str) -> int:
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

print('------------------------------------------------')
print('First Task')
print(count_vowels("HelloWorld"))  # 3
print(count_vowels("Python"))      # 1

# Задача №2
# Написать функцию на вход, которой подается строка. Функция должна вернуть true, если каждый символ в строке встречается только 1 раз, иначе должна вернуть false.
def all_unique(s: str) -> bool:
    return len(s) == len(set(s))

print('------------------------------------------------')
print('2 Task')
print(all_unique("abcd"))   #  True (все символы разные)
print(all_unique("hello"))  #  False (буква 'l' повторяется)

# Задача №3
# Написать функцию, которая принимает положительное число и возвращает количество бит равных 1 в этом числе.

def count_bits(n: int) -> int:
    if n < 0:
        raise ValueError("Число должно быть положительным")
    return bin(n).count("1")

print('------------------------------------------------')
print('3 Task')
print(count_bits(0))   # 0 -> 0
print(count_bits(7))   # 7 -> 111 => 3
print(count_bits(9))   # 9 -> 1001 => 2
print(count_bits(255)) # 255 -> 11111111 => 8
# Задача №4
# Написать функцию, которая принимает положительное. Функция должна вернуть то, сколько раз необходимо перемножать цифры числа или результат перемножения, чтобы получилось число состоящее из одной цифры.
# Например, для входного числа:
# ·       39 функция должна вернуть 3, так как 3*9=27 => 2*7=14 => 1*4=4
# ·       4 функция должна вернуть 0, так как число уже состоит из одной цифры
# ·       999 функция должна вернуть 4, так как 9*9*9=729 => 7*2*9=126 => 1*2*6=12 => 1*2=2

def multiplicative_persistence(n: int) -> int:
    if n < 0:
        raise ValueError("Число должно быть положительным")

    steps = 0
    while n >= 10:  # пока число состоит более чем из одной цифры
        product = 1
        for digit in str(n):
            product *= int(digit)
        n = product
        steps += 1
    return steps

print('------------------------------------------------')
print('4 Task')
print(multiplicative_persistence(39))   # 3 → 3*9=27 → 2*7=14 → 1*4=4
print(multiplicative_persistence(4))    # 0 (уже одна цифра)
print(multiplicative_persistence(999))  # 4 → 999→729→126→12→2
print(multiplicative_persistence(25))   # 2 → 2*5=10 → 1*0=0

# Задача №5
# Написать функция, которая принимает два целочисленных вектора одинаковой длины и возвращает среднеквадратическое отклонение двух векторов.

from math import sqrt

def rmse(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Векторы должны быть одинаковой ненулевой длины")

    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))

print('------------------------------------------------')
print('5 Task')
print(rmse([1, 2, 3], [1, 2, 4]))  # 0.5773502691896257
print(rmse([10, 20, 30], [10, 20, 30]))  # 0.0
print(rmse([2, 4, 6], [1, 3, 5]))  # 1.0

# Задача №6
# Написать функцию, которая принимает список чисел и возвращает кортеж из двух элементов. Первый элемент кортежа – мат. ожидание, второй элемент – СКО. Запрещается использовать функции для расчета соответствующих характеристик.
from math import sqrt
from typing import List, Tuple

def mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        raise ValueError("Список не должен быть пустым")

    n = len(xs)
    mean = sum(xs) / n  # математическое ожидание
    variance = sum((x - mean) ** 2 for x in xs) / n  # дисперсия
    std_dev = sqrt(variance)  # стандартное отклонение (СКО)

    return mean, std_dev

print('------------------------------------------------')
print('6 Task')
print(mean_std([1, 2, 3, 4, 5]))   # (3.0, 1.4142135623730951)
print(mean_std([10, 10, 10]))      # (10.0, 0.0)
print(mean_std([2, 4, 6, 8]))      # (5.0, 2.23606797749979)
# Задача №7
# Написать функцию, принимающая целое положительное число. Функция должна вернуть строку вида “(n1**p1)(n2**p2)…(nk**pk)” представляющая разложение числа на простые множители (если pi == 1, то выводить только ni).
def prime_factors_fmt(n: int) -> str:
    if n < 2:
        return f"({n})"

    result = []
    divisor = 2

    while divisor * divisor <= n:
        power = 0
        while n % divisor == 0:
            n //= divisor
            power += 1
        if power == 1:
            result.append(f"({divisor})")
        elif power > 1:
            result.append(f"({divisor}**{power})")
        divisor += 1 if divisor == 2 else 2  # после 2 проверяем только нечётные числа

    # если после деления остался простой множитель больше 1
    if n > 1:
        result.append(f"({n})")

    return "".join(result)

print('------------------------------------------------')
print('7 Task')
print(prime_factors_fmt(86240))  # (2**5)(5)(7**2)(11)
print(prime_factors_fmt(360))    # (2**3)(3**2)(5)
print(prime_factors_fmt(17))     # (17)
print(prime_factors_fmt(1001))   # (7)(11)(13)
# Например, для числа 86240 функция должна вернуть “(2**5)(5)(7**2)(11)”
# Задача №8
# Написать функцию, принимающая 2 строки вида “xxx.xxx.xxx.xxx” представляющие ip-адрес и маску сети. Функция должна вернуть 2 строки: адрес сети и широковещательный адрес.

def ip_calc_network_broadcast(ip: str, mask: str) -> tuple[str, str]:
    def to_int(addr: str) -> int:
        """Преобразует IP строку в 32-битное целое число"""
        parts = addr.split(".")
        if len(parts) != 4:
            raise ValueError("Неверный формат IP-адреса")
        nums = [int(p) for p in parts]
        if any(not (0 <= n <= 255) for n in nums):
            raise ValueError("Октеты должны быть в диапазоне 0..255")
        return (nums[0] << 24) | (nums[1] << 16) | (nums[2] << 8) | nums[3]

    def to_str(value: int) -> str:
        """Преобразует 32-битное число обратно в IP строку"""
        return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))

    ip_int = to_int(ip)
    mask_int = to_int(mask)

    network = ip_int & mask_int
    broadcast = network | (~mask_int & 0xFFFFFFFF)

    return to_str(network), to_str(broadcast)

print('------------------------------------------------')
print('8 Task')
print(ip_calc_network_broadcast("192.168.1.10", "255.255.255.0"))
# ('192.168.1.0', '192.168.1.255')

print(ip_calc_network_broadcast("10.0.5.25", "255.255.255.192"))
# ('10.0.5.0', '10.0.5.63')

print(ip_calc_network_broadcast("172.16.10.123", "255.255.255.240"))
# ('172.16.10.112', '172.16.10.127')

# Задача №9
# Написать функцию, принимающая целое число n, задающее количество кубиков. Функция должна определить, можно ли из данного кол-ва кубиков построить пирамиду, то есть можно ли представить число n как 1^2+2^2+3^2+…+k^2. Если можно, то функция должна вернуть k, иначе строку “It is impossible”.
def cube_pyramid(n: int):
    if n <= 0:
        raise ValueError("Число должно быть положительным")

    total = 0
    k = 0
    while total < n:
        k += 1
        total += k ** 2
        if total == n:
            return k
    return "It is impossible"

print('------------------------------------------------')
print('9 Task')
print(cube_pyramid(1))      # 1  -> 1^2
print(cube_pyramid(5))      # 2  -> 1^2 + 2^2 = 5
print(cube_pyramid(14))     # 3  -> 1^2 + 2^2 + 3^2 = 14
print(cube_pyramid(30))     # It is impossible
print(cube_pyramid(55))     # 5  -> 1^2+2^2+3^2+4^2+5^2=55

# Задача №10
# Написать функцию, которая принимает положительное целое число n и определяющая является ли число n сбалансированным. Число является сбалансированным, если сумма цифр до средних цифр равна сумме цифр после средней цифры. Если число нечетное, то средняя цифра одна, если четное, то средних цифр две. При расчете, средние числа не участвуют.
# Например:
# ·       Число 23441 сбалансированное, так как 2+3=4+1
# ·       Число 7 сбалансированное, так как 0=0
# ·       Число 1231 сбалансированное, так как 1=1
# ·       Число 123456 несбалансированное, так как 1+2!=5+6

def is_balanced_number(n: int) -> bool:
    s = str(n)
    length = len(s)
    mid = length // 2

    if length == 1:
        return True  # одна цифра всегда сбалансирована

    if length % 2 == 0:
        left = s[:mid - 1]
        right = s[mid + 1:]
    else:
        left = s[:mid]
        right = s[mid + 1:]

    sum_left = sum(map(int, left))
    sum_right = sum(map(int, right))

    return sum_left == sum_right

print('------------------------------------------------')
print('10 Task')
print(is_balanced_number(23441))   # True (2+3 = 4+1)
print(is_balanced_number(7))       # True (0=0)
print(is_balanced_number(1231))    # True (1=1)
print(is_balanced_number(123456))  # False (1+2 != 5+6)
print(is_balanced_number(123321))  # True (1+2+3 = 3+2+1)