from shared import save_data
import numpy as np
import time

eps = 1e-5

def sigmoid(x):
    """
    Assume x is a numpy array
    """
    # assert (-746 < x & x < 746).all()
    x[x > 500] = 500
    x[x < -500] = -500
    ans = np.choose(x<0,
                    (1/(1+np.exp(-x)),
                     np.exp(x)/(1+np.exp(x)))
    )
    assert np.isfinite(ans).all()
    if not (ans!=0).all():
        print(x.min())
    assert (ans != 0).all()
    return ans

def log_cost(X, y, phi):
    n,d = X.shape
    assert y.shape == (n,1)
    assert phi.shape == (d,1)
    
    """
    X is our data points
    y is in {0,1}
    theta is our current weight
    This function computes the negative log likelihood
    of the current parameters
    """
    sum = np.sum(np.choose(y==1,(np.log(sigmoid(-(X @ phi))),np.log(sigmoid(X @ phi)))))
    assert np.isfinite(sum)
    return -sum/n

#def cost_except_one_dimension(phi, dimension,

def log_grad(X, y, phi):
    n,d = X.shape
    assert y.shape == (n,1)
    assert phi.shape == (d,1)

    return -X.T @ ( y - sigmoid(X @ phi)) / n

def numeric_grad(X, y, phi):
    n,d = X.shape
    assert y.shape == (n,1)
    assert phi.shape == (d,1)


def binary_error(X,y,phi):
    return np.sum(classify(phi,X) != (y==1))

def learn_stochastic(X, y):
    n,d = X.shape
    assert y.shape == (n,1)
    
    tired = False
    print(d)
    #phi = np.random.uniform(size=(d))
    phi = np.zeros((d,1))
    eta = 0.1
    prev_cost = log_cost(X,y,phi)

    t0 = time.time()
    k = 100
    while not tired:
        for _ in range(50000//k):
            sample = np.random.choice(n,k)
            tmpX = X[sample]
            tmpy = y[sample]
            gradient = log_grad(tmpX, tmpy, phi)
            phi_new = phi - eta * gradient
            phi = phi_new

        E_in_binary = binary_error(X,y,phi)
        print("[%7.2f] %4d %7.4e %20.10f %10.5f" % (time.time()-t0, E_in_binary, eta, log_cost(X, y, phi), np.sqrt(np.sum(phi**2))))
        if E_in_binary == 0:
            tired = True

    print(log_cost(X,y,phi))
    return phi

def learn(X, y):
    n,d = X.shape
    assert y.shape == (n,1)
    
    tired = False
    print(d)
    #phi = np.random.uniform(size=(d))
    phi = np.zeros((d,1))
    eta = 0.1
    prev_cost = log_cost(X,y,phi)
    while not tired:
        E_in_binary = binary_error(X,y,phi)
        print("%4d %7.4e %20.10f %10g" % (E_in_binary, eta, log_cost(X,y,phi), np.sqrt(np.sum(phi**2))))
        gradient = log_grad(X, y, phi)
        phi_new = phi - eta * gradient
        #phi = phi_new
        new_cost = log_cost(X,y,phi_new)
        if prev_cost > new_cost:
            prev_cost = new_cost
            phi = phi_new
            eta = eta*1.1
        else:
            eta = eta/1.1
        
        if np.sqrt(np.sum(gradient*gradient)) < eps/1e3 or E_in_binary == 0:
            tired = True

    print(log_cost(X,y,phi))
    return phi

def classify(phi, image):
    return image @ phi >= 0

def save_parameters(theta):
    save_data('params', theta=best_theta)

def main():   
    pass

if __name__ == "__main__":
    main()
