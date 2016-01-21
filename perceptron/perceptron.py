from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time

def find_missclassified(points, labels, sol):
    points = np.asarray(points)
    labels = np.asarray(labels)
    sol = np.asarray(sol)
    
    n,d = points.shape
    d, = sol.shape

    sol_labels = (points @ sol) > 0
    sol_labels = sol_labels*2 - 1

    diff  = (labels != sol_labels)
    diff, = diff.nonzero()
    if len(diff) == 0:
        return None
    idx = np.random.choice(diff)
    #print(idx)
    return idx
    
def perceptron(ax, points, labels):
    n,d = points.shape # n x d matrix
    curr = np.zeros(d).astype(np.float)
    tired = False
    while not tired:
        #show(points,labels,curr)
        idx = find_missclassified(points, labels, curr)
        if idx is None:
            break
        curr += labels[idx]*points[idx] # 
    return curr

def show(points, labels, sol):
    ax=plt.axes()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    _,xs,ys = points.T
    ax.plot(xs[labels==1],ys[labels==1],'r+')
    ax.plot(xs[labels==-1],ys[labels==-1],'g+')
    c,a,b = sol
    if (b != 0):
        ax.plot([0,1],[-c/b,(-a-c)/b])
    plt.show()
    
def main():
    ax = plt.axes()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    c = np.random.uniform(-1,0)
    a = np.random.uniform(-1-c, -c)
    b = 1
    target = np.array([c,a,b]).reshape(3,1)
    ax.plot([0,1],[-c/b,(-a-c)/b],'k')
    n = 100000 # n points
    points = np.random.uniform(size=(n,2))
    points = np.c_[np.ones((n,1)), points]
    labels = (points @ target) > 0
    labels = labels[:, 0]*2 - 1
    _,xs,ys = points.T
    ax.plot(xs[labels==1],ys[labels==1],'r+')
    ax.plot(xs[labels==-1],ys[labels==-1],'g+')
    t1 = time.time()
    sol = perceptron(ax, points, labels)
    t2 = time.time()
    print("diff: %s"%(t2-t1))
    c,a,b = sol
    ax.plot([0,1],[-c/b,(-a-c)/b],'b')
    linreg = np.linalg.pinv(points) @ labels.reshape((n,1))
    c,a,b = linreg
    ax.plot([0,1],[-c/b,(-a-c)/b],'g')
    plt.show()
    
if __name__ == "__main__":
    main()

