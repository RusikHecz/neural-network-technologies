import numpy as np 
import random
import math
import matplotlib.pyplot as plt

def func(i):
    return ((i % 20) + 1) / 20

def gen_sequence(seq_len = 1000):
    seq = [abs(math.sin(i/20)) + func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)),seq)
    plt.show()

draw_sequence()