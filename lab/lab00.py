import sys

import numpy as np

sys.path.append("../")
from utils import *



def create_cartesian_product(vec1, vec2):
    pass


def find_closest(a, n):
    a = np.array(a)
    return np.min(np.abs(a-n))

print(find_closest([1, 24, 12, 13, 14], 10))