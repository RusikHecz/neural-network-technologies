import gens
import numpy as np


def random_line(img_size=50):
    p = np.random.random([1])
    if p < 0.5:
        return gens.gen_v_line(img_size)
    else:
        return gens.gen_h_line(img_size)


def gen_k_lines(k, img_size=50):
    img = np.zeros([img_size, img_size])
    for i in range(k):
        img += random_line(img_size)
    img[np.nonzero(img)] = 1
    return img


def gen_data(size=500, img_size=50):
    c1 = size // 3
    c2 = c1
    c3 = size - (c1 + c2)

    label_c1 = np.full([c1, 1], 'One')
    data_c1 = np.array([gen_k_lines(1, img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Two')
    data_c2 = np.array([gen_k_lines(2, img_size) for i in range(c2)])
    label_c3 = np.full([c3, 1], 'Three')
    data_c3 = np.array([gen_k_lines(3, img_size) for i in range(c3)])

    data = np.vstack((data_c1, data_c2, data_c3))
    label = np.vstack((label_c1, label_c2, label_c3))

    return data, label

