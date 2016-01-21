from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time

def experiment():
    # 0 = tails, 1 = heads
    mat = np.random.randint(0, 2, (1000,10))
    c_1 = 0
    c_rand = np.random.choice(mat.shape[0])
    c_min = mat.sum(axis=1).argmin()
    random_row = mat[c_rand,:]

    v_1 = (mat[0,:] == 1).sum() / 10.0
    v_rand = (mat[c_rand,:] == 1).sum() / 10.0
    v_min = (mat[c_min,:] == 1).sum() / 10.0
    return ([c_1, c_rand, c_min],[v_1, v_rand, v_min])

def main():
    """Solves exercise 1.10 from the book
    "Learning from data"
    """
    # random 1000x10 0/1 matrix
    data = [experiment() for x in range(10000)]
    c_1s = []
    c_rands = []
    c_mins = []

    v_1s = []
    v_rands = []
    v_mins = []

    for a,b in data:
        c_1, c_rand, c_min = a;
        v_1, v_rand, v_min = b;
        c_1s.append(c_1)
        c_rands.append(c_rand)
        c_mins.append(c_min)
        v_1s.append(v_1)
        v_rands.append(v_rand)
        v_mins.append(v_min)

    data_c1 = np.asarray(c_1s)
    data_crand = np.asarray(c_rands)
    data_cmin = np.asarray(c_mins)
    
    data_v1 = np.asarray(v_1s)
    data_vrand = np.asarray(v_rands)
    data_vmin = np.asarray(v_mins)

    hist_r,bins_r = np.histogram(data_vrand)
    hist_m,bins_m = np.histogram(data_vmin)
    center = (bins_r[:-1]+bins_r[1:]) / 2.0
    width = 0.7 * (bins_r[1] - bins_r[0])
    print(center)
    plt.figure(1)
    plt.subplot(221)
    plt.bar(center, hist_r, align="center", width=0.1)

    plt.subplot(223)
    center = (bins_m[:-1]+bins_m[1:]) / 2.0
    width = 0.7 * (bins_r[1] - bins_r[0])
    plt.bar(center, hist_m, align="center", width=0.1)
    print("hist_r: %s\nbins_r: %s"%(hist_r,bins_r))
    print("hist_m: %s\nbins_m: %s"%(hist_m,bins_m))


    # expected value of min is approximately 1-(1-1/2^10)^1000
    
    u_1 = 0.5
    u_r = 0.5
    u_min = (1-(1/2**10)**1000)/10

    n = 10
    xs = np.linspace(0,1,100)
    hoeffding = 2*np.exp((xs**2)*n*2*(-1))
    hoeffding = [min(h,1.0) for h in hoeffding]
    plt.subplot(222)
    print(hoeffding)
    tmp = [np.absolute(data_vrand - u_r )> x for x in xs]
    tmp = [t.sum()/100000 for t in tmp]
    plt.plot(xs, hoeffding, 'bo', xs, tmp, 'r+')
    
    plt.subplot(224)
    tmp = [np.absolute(data_vmin - u_min) > x for x in xs]
    tmp = [t.sum()/100000 for t in tmp]
    plt.plot(xs, hoeffding, 'bo', xs, tmp, 'r+')
    
    plt.show()    
if __name__ == "__main__":
    main()
