from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time

def f(x):
    return x*x

def experiment(N):
    N_prime = 1000
    points = np.linspace(-1,1,N_prime).reshape((N_prime, 1))
    x1 = np.random.uniform(-1,1,(N, 1))
    x2 = np.random.uniform(-1,1,(N, 1))
    y1 = x1**2
    y2 = x2**2
    As = (y1-y2)/(x1-x2)  # N x 1
    Bs = y1-As*x1  # N x 1
    prediction = (As @ points.T) + Bs  # N x N'
    actual = (points**2).reshape((1, N_prime))  # 1 x N'
    diff = prediction - actual  # N x N'
    Eout = (diff ** 2).mean(axis=1)  # N
    return As, Bs, Eout

def main():
    NN = 1000000
    N = 100000
    As, Bs, Eout = tuple(
        np.concatenate(xs)
        for xs in zip(*[experiment(N) for _ in range(NN // N)]))
    print(Eout.shape) 
    print("np.mean(As) = %s"%np.mean(As))
    print("np.mean(Bs) = %s"%np.mean(Bs))
    print("np.mean(Eout) = %s"%np.mean(Eout))
    print("np.var(Eout) = %s" % np.std(Eout)**2)
    Eout.sort()
    plt.plot(np.arange(NN) + 1, Eout, 'o')
    plt.plot((1, NN), (8/15, 8/15), '-')
    plt.show()
if __name__ == "__main__":
    main()
