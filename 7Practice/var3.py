import numpy as np 
import random
import math
import matplotlib.pyplot as plt

def func(i):
    i = i % 21
    return abs(i - 10)/4

def gen_sequence(seq_len = 1000):
    seq = [math.cos(i/4) + func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)),seq)
    plt.show()
    
draw_sequence()