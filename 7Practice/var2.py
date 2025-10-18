import numpy as np 
import random
import math
import matplotlib.pyplot as plt


def gen_sequence(seq_len = 1000):
    seq = [math.sin(i/5)*math.sin(i/10+0.5) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(100)
    plt.plot(range(len(seq)),seq)
    plt.show()
    
draw_sequence()