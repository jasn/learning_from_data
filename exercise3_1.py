from __future__ import division
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def pocket(x,y):
    iterations = 0
    w = np.random.random_sample(x.shape[1]) #init to zero
    best_w = w
    best_error = x.shape[0]
    tired = False
    data = []
    first = True
    while not tired:
        iterations = iterations + 1
        if iterations == 100000:
            tired = True
        res = (x @ w.T * y)
        tmp = np.where(res<0)[0]
        if len(tmp) == 0 and not first:
            best_w = w
            data.append(0)
            break
        first = False
        idx = np.random.choice(tmp)
        w = w+x[idx]*y[idx]
        error = len(np.where((x @ w.T * y)<0)[0])
        if error < best_error:
            best_w = w
            best_error = error
        data.append(best_error)
    data = np.asarray(data)
    return best_w, data

def PLA(x,y):
    iterations = 0
    # x has d+1 dimensions, with dimension d+1 set to 1 always
    # y is +1/-1
    w = np.asarray([0,0,1]) # x1 x2 x_{d+1}
    tired = False
    while not tired:
        res = (x @ w.T * y)
        tmp = np.where(res<0)[0]
        if len(tmp) == 0:
            break
        idx = tmp[0]
        w = w+x[idx]*y[idx]
        iterations = iterations+1
    return w,iterations

def linreg(x,y):
    return np.linalg.pinv(x) @ y

def data(thk,rad,sep,N):
    thk,rad,sep,N = (np.float(thk),np.float(rad),np.float(sep),np.int(N))
    red_center_x = 0
    red_center_y = 0
    blue_center_x = rad+thk/2.0
    blue_center_y = -sep

    a2=rad**2 # inner ring
    b2=(rad+thk)**2 # outer ring
    r2 = (b2-a2)*np.random.random_sample(N)+a2 # sample between a and b
    angle = np.pi*np.random.random_sample(N)
    blue_red = np.random.choice((-1,1), N) # -1 is red, 1 is blue
    xs = np.sqrt(r2)*np.cos(angle)
    xs = xs + blue_center_x/2.0
    xs = xs + blue_red*blue_center_x/2.0
    ys = np.sqrt(r2)*np.sin(angle)
    ys = ys*blue_red*(-1)+blue_center_y/2.0
    ys = ys+blue_red*blue_center_y/2.0
    zs = np.ones(N)
    X = np.array([xs,ys,zs]).T

    return X,blue_red,xs,ys

def doPlot(xs,ys,lines):
    plt.plot(xs,ys,'o',color='b')

    for (a,b,c) in lines:
        x = [-15,30]
        y = [(-c-a*v)/b for v in x]
        plt.plot(x,y,color='r')
   
def main(thk, rad, sep, N, plot=True, doLinreg=True):
    thk,rad,sep,N = (np.float(thk),np.float(rad),np.float(sep),np.int(N))
    X,blue_red,xs,ys = data(thk,rad,sep,N)
    (a,b,c),iterations=PLA(X,blue_red)
    #print("a=%s\nb=%s\nc=%s"%(a,b,c))
    x = [-rad-thk-1,rad+rad+thk+1]
    y = [(-c-a*v)/b for v in x]
    if plot is True:
        plt.plot(x,y,'-',color='g')

    if doLinreg is True:
        a,b,c = linreg(X,blue_red)
        x = [-rad-thk-1,rad+rad+thk+1]
        y = [(-c-a*v)/b for v in x]
        if plot is True:
            plt.plot(x,y,'-',color='r')

    if plot is True:
        plt.plot(xs,ys,'o',color='b')
        plt.show()

    return iterations

def dth_order_monomials(d):
    d=d+1
    monomials = [(i,j) for i in range(d) for j in range(d-i)]
    return monomials

def transform_dth_order(X,d):
    monomials = dth_order_monomials(d)
    xs = X.T[0]
    ys = X.T[1]
    final = []
    for (i,j) in monomials:
        final.append(xs**i * ys**j)
    return np.asarray(final).T
    

def transform_3rd_order(X):
    return transform_dth_order(X,3)

def exercise3_2(thk,rad,N=2000):
    N=np.int(N)
    tests = 20
    seps = np.linspace(0.05,0.5,25) # [0.2,0.4,...,5.0]
    avgs = [np.average([main(thk,rad,sep,N,False,False)
                        for x in np.arange(tests)]) for sep in seps]

    plt.plot(seps,avgs,'o',color='b')
    plt.show()
    
def exercise3_3(thk,rad,sep,N=2000):
    N=np.int(N)
    X,blue_red,xs,ys = data(thk,rad,sep,N)
    final,d = pocket(X,blue_red)
    d = d / len(d)
    doPlot(xs,ys,[final])    
    plt.show()
    #plt.plot(np.arange(len(d)),d,'o')
    #plt.show()

def pltContour(w, xlim, ylim):
    x1,x2 = xlim
    y1,y2 = ylim
    x,y = np.mgrid[x1:x2:100j,y1:y2:100j]
    monomials = dth_order_monomials(3)
    z = sum(c*x**i*y**j for c,(i,j) in zip(w, monomials))
    plt.contour(x,y,z, [0], color='g')
    
def exercise3_3e(thk,rad,sep,N=2000):
    N=np.int(N)
    X,blue_red,xs,ys = data(thk,rad,sep,N)
    X=transform_3rd_order(X)
    final,d = pocket(X,blue_red)
    d = d / len(d)
    plt.subplot(121)
    plt.plot(np.arange(len(d)),d,'o')
    plt.subplot(122)
    x, y = np.mgrid[-15:30:100j,-15:30:100j]
    monomials = dth_order_monomials(3)
    z = sum(c*x**i*y**j for c,(i,j) in zip(final, monomials))
    plt.contour(x,y,z, [0], color='g')

    w = linreg(X,blue_red)
    z = sum(c*x**i*y**j for c,(i,j) in zip(w, monomials))
    plt.contour(x,y,z, [0], color='r')
    plt.plot(xs,ys,'+',color='b')
    
    plt.show()
  
def exercise3_7(thk, rad, sep, N=2000):
    N = np.int(N)
    X,blue_red,xs,ys = data(thk,rad,sep,N)
    X=transform_3rd_order(X)
    d = X.shape[1]
    A = -X*(blue_red.reshape((N,1)))
    # A[:, 2] = np.ones(N)
    I = np.identity(N)
    A = np.concatenate((A,-I), axis=1)
    b = -np.ones(N)
    c = np.concatenate((np.zeros(d),np.ones(N)))

    bounds_w = tuple((None,None) for i in range(d)) # -inf < x_n < inf
    #bounds_y = ((-1,-1),)
    #bounds_z = ((None,None),)
    bounds_err = tuple((0,None) for i in range(N)) # eps_n >= 0
    bounds_all = bounds_w+bounds_err

    def cb(xk, **kwargs):
        sys.stdout.write("\rphase %s #%s"%(kwargs["phase"],kwargs["nit"]))
        sys.stdout.flush()

    res = opt.linprog(c,A_ub=A,b_ub=b,bounds=bounds_all, callback=cb,options=dict(maxiter=2000,tol=1e-6))
    print("")
    #print(res)

    print(res.message)
    if res.status in (0,1):
    #coef,slack,success,status = res.x,res.slack,res.success,res.status
        coef = res.x
        print(coef)
        if d > 3:
            doPlot(xs,ys,[])
            pltContour(coef[0:d],plt.xlim(),plt.ylim())
            plt.show()
            pass
        else:
            doPlot(xs,ys,[coef[0:3].tolist()])
            plt.show()
    else:
        #print(res.status)
        doPlot(xs,ys,[(0.1,-1,-2.5)])
        plt.show()
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Expected 31, 32, or 33")
        sys.exit(1)
    if sys.argv[1] == "31":
        # thk, rad, sep, N as double, double, double, integer
        print("Expected 4 parameters: thk, rad, sep, N as illustrated in the exercise")
        print("Received parameters:")
        _,_,thk,rad,sep,N = sys.argv
        print("thk:%s"%thk)
        print("rad:%s"%rad)
        print("sep:%s"%sep)
        print("N:%s"%N)
        main(thk,rad,sep,N)
        sys.exit(0)
        
    if sys.argv[1] == "32":
        _,_,thk,rad,sep,N = sys.argv
        exercise3_2(thk,rad,N)
        sys.exit(0)
    if sys.argv[1] == "33":
        _,_,thk,rad,sep,N = sys.argv
        exercise3_3(thk,rad,sep,N)
    if sys.argv[1] == "33e":
        _,_,thk,rad,sep,N = sys.argv
        exercise3_3e(thk,rad,sep,N)
    if sys.argv[1] == "37":
        _,_,thk,rad,sep,N = sys.argv
        while True:
            exercise3_7(thk,rad,sep,N)
