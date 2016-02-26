from main import log_cost, log_grad
import shared
import numpy as np
from math import sqrt

def test_cost(X,y):
    tired = False
    eps = 0.01
    counter = 0
    while not tired and counter < 20:
        counter = counter + 1
        phi=np.random.uniform(size=(X.shape[1],1))
        print(log_cost(X, y, phi))

def test_gradient(X,y):
    tired = False
    eps = 0.01
    counter = 0
    d = X.shape[1]
    while not tired and counter < 100:
        counter = counter + 1
        phi=np.random.uniform(size=(d,1))
        log_grad_formula = log_grad(X,y,phi)
        log_grad_formula_norm = np.sqrt(np.sum(log_grad_formula**2))
        log_grad_numerical = (log_cost(X,y,phi+eps) - log_cost(X,y,phi-eps))/(2*eps*sqrt(d))
        log_grad_numerical_norm = np.sqrt(np.sum(log_grad_numerical**2))
        diff = np.sqrt(np.sum((log_grad_formula-log_grad_numerical)**2))
        print("||log_grad(phi)|| = %s"%log_grad_formula_norm)
        print("||approximated||  = %s"%log_grad_numerical_norm)
        print("||diff|| = %s"%(diff))



if __name__ == "__main__":
    small_data = shared.load_small_data()
    X = small_data['images']
    y = np.concatenate((np.ones(X.shape[0]/2), np.zeros(X.shape[0]/2)))
    y = y.reshape(-1,1)
    test_cost(X, y)
    test_gradient(X, y)

