import numpy as np

def boostrap_sample(set_size, X, y = None):
    indices = np.random.randint(0, X.shape[0], size = set_size)
    X_boostrap = X[indices]
    if y is not None:
        return X_boostrap, y[indices]
    return X_bootstrap

def sample_random_features(num_features, X):
    features = np.random.choice(X.shape[1], size = num_features, replace = False)
    X_subset = X[:,features]
    return X_subset, features
