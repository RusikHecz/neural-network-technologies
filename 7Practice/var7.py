import numpy as np 
import random
import math
import matplotlib.pyplot as plt

def func(i):
    i = abs((i/2 % 27) - 13)
    return (math.log(i+1))/2 - 0.5

def gen_sequence(seq_len = 1000):
    seq = [ func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)),seq)
    plt.show()

draw_sequence()