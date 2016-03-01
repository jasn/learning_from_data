from shared import save_data
import numpy as np
import time

eps = 1e-5
"""
In project we apply machine learning algorithms for digit recognition.
The data is a set of images labelled with a number between 0 and 9, which it is supposed to look like.
We want to accomplish two tasks:
1) We want to use logistic regression for seperating all pairs of digits.
i.e. we need to train a classifier for each pair of labels.
2) We want to generalize logistic regression to 'softmax' which has k classes, rather than two.

In this file the gradient descent algorithm is implemented to optimize the negative log-likelihood
function for logistic regression.

We provide both a stochastic gradient descent and a deterministic gradient descent algorithm.
The stochastic gradient descent is implemented by taking a small random subset of the input, and apply
normal gradient descent on the subset.
"""

    
def tired():
    tired.iteration += 1
    if tired.iteration < tired.max_iterations:
        return False
    tired.iteration = 0
    return True
tired.max_iterations = 100
tired.iteration = 0

def softmax(x):
    """
    Assume x is a numpy array.
    softmax(x) is used for k-wise classifiers.
    """
    c = np.max(x)
    normalization_factor = np.log(c + np.log(np.sum(x-c)))
    return np.exp(x-normalization_factor)

def sigmoid(x):
    """
    Assume x is a numpy array.
    sigmoid(x) is used for logistic regression.
    """
    assert np.isfinite(x).all()
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

def regularize(val, phi, l=0.0):
    return val + l*np.concatenate((phi[:-1],[[0]]))

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

def log_grad(X, y, phi):
    n,d = X.shape
    assert y.shape == (n,1)
    assert phi.shape == (d,1)
    assert np.isfinite(phi).all()
    assert np.isfinite(-X.T @ ( y - sigmoid(X @ phi)) / n).all()
    
    return -X.T @ ( y - sigmoid(X @ phi)) / n

def numeric_grad(X, y, phi):
    n,d = X.shape
    assert y.shape == (n,1)
    assert phi.shape == (d,1)


def binary_error(X,y,phi):
    return np.sum(classify(phi,X) != (y==1))

def learn_stochastic(X, y, l=0.0):
    n,d = X.shape
    assert y.shape == (n,1)
    
    #phi = np.random.uniform(size=(d))
    phi = np.zeros((d,1))
    eta = 0.1
    prev_cost = log_cost(X,y,phi)

    t0 = time.time()
    k = 10
    zero_error = False
    for _ in range(k): #while not tired() and not zero_error:
        for __ in range(50000//k):
            sample = np.random.choice(n,k)
            tmpX = X[sample]
            tmpy = y[sample]
            try:                
                gradient = regularize(log_grad(tmpX, tmpy, phi), phi, l)
            except Exception:
                print(_,__)
                raise

            phi_new = phi - eta * gradient
            phi = phi_new

        E_in_binary = binary_error(X,y,phi)
        print("[%7.2f] %4d %7.4e %20.10f %10.5f" % (time.time()-t0, E_in_binary, eta, log_cost(X, y, phi), np.sqrt(np.sum(phi**2))))
        if E_in_binary == 0:
            zero_error = True
            break

    print(log_cost(X,y,phi))
    return phi

def learn(X, y, l):
    n,d = X.shape
    assert y.shape == (n,1)
    
    print(d)
    #phi = np.random.uniform(size=(d))
    phi = np.zeros((d,1))
    eta = 0.1
    prev_cost = log_cost(X,y,phi)
    zero_error = False
    while not tired() and not zero_error:
        E_in_binary = binary_error(X, y, phi)
        print("%4d %7.4e %20.10f %10g" % (E_in_binary, eta, log_cost(X,y,phi), np.sqrt(np.sum(phi**2))))
        gradient = regularize(log_grad(X, y, phi), phi, l)
        phi_new = phi - eta * gradient
        #phi = phi_new
        new_cost = log_cost(X,y,phi_new)
        if prev_cost > new_cost:
            prev_cost = new_cost
            phi = phi_new
            eta = eta*1.1
        else:
            eta = eta/1.1
        
        if E_in_binary == 0:
            zero_error = True

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
