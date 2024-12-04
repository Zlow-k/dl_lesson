import pickle
from sklearn.model_selection import train_test_split

def load_mnist(path):
    with open(f'{path}/mnist_X.pkl', 'rb') as f:
        X = pickle.load(f)      
    with open(f'{path}/mnist_y.pkl', 'rb') as f:
        y = pickle.load(f)
    X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=10000)
    
    return X_train, X_test, t_train, t_test