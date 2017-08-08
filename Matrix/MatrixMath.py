import numpy as np
import sys

def is_singular(mat):
    try:
        np.linalg.inv(mat)
    except:
        return False
    return not (np.linalg.cond(mat) < 1.0/sys.float_info.epsilon)
