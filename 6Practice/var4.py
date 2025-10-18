import gens
import numpy as np


def random_line(img_size=50):
    p = np.random.random([1])
    if p < 0.5:
        return gens.gen_v_line(img_size)
    else:
        return gens.gen_h_line(img_size)


def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1

    label_c1 = np.full([c1, 1], 'Cross')
    data_c1 = np.array([gens.gen_cross(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Line')
    data_c2 = np.array([random_line(img_size) for i in range(c2)])

    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))

    return data, label
