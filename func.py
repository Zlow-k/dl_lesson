import numpy as np

def step_func(x):
    y = x > 0
    return y.astype(int)

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def relu(x):
    return np.maximum(0, x)

def identity_func(x):
    return x

def softmax(a):
    # オーバーフローの恐れあり
    # y = np.exp(a) / np.sum(np.exp(a))
    
    c = np.max(a)
    y = np.exp(a-c) / np.sum(np.exp(a-c))
    return y