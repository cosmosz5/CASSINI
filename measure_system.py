from scipy.linalg import dft
import numpy as np
import pdb

def uv_discrete():



    return 0

def measure_system(size, uv_dis):
    m = dft(size)
    system = np.dot(uv_dis, m)

    return system

