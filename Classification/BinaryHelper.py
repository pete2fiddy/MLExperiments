import numpy as np

def modify_y_to_positive_y_val(y, target_y_val):
    true_indexes = np.where(y == target_y_val)
    y_out = np.zeros(y.shape)
    y_out[true_indexes] = 1.0
    return y_out

def modify_y_to_range(y, y_is_pos_val, pos_val, neg_val):
    modified_y = modify_y_to_positive_y_val(y, y_is_pos_val)
    indices_y_pos = np.where(modified_y == 1.0)
    indices_y_neg = np.where(modified_y != 1.0)
    modified_y[indices_y_pos[0]] = pos_val
    modified_y[indices_y_neg[0]] = neg_val
    return modified_y
