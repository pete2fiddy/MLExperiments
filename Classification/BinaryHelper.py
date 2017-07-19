import numpy as np

def modify_y_to_positive_y_val(y, target_y_val):
    true_indexes = np.where(y == target_y_val)
    print("y: ", y)
    y_out = np.zeros(y.shape)
    y_out[true_indexes] = 1.0
    return y_out
