import numpy as np
import time
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


def backpropagation_iteration(x, y, activations, weights, bias):
    """
    Returns the differential qoutient of each
    edge in the neural network.
    Formally, between layers l and l+1 we compute
    dz^{l+1}_i/dz^{l}_j, since that allows us to compute
    d E_in / d z^{l}_{i}, which in turn allows us to
    compute d E_in / d w_{i,j}^l and d E_in / d b_i^l
    """
    d, = x.shape
    assert y.shape == (weights[-1].shape[0],)
    assert len(activations) == len(weights)+1 == len(bias)+1
    assert all(a.shape[0] == w.shape[1] for a, w in zip(activations[:-1], weights))
    assert all(a.shape[0] == w.shape[0] for a, w in zip(activations[1:], weights))
    
    output = activations[-1]

    assert output.shape == y.reshape(-1,1).shape
    
    dz = -(y.reshape(-1,1) - output)*output*(1-output) # column vector of shape (n_L x 1)
    deltas = [dz]
    prev = dz # invariant: prev.shape = (n_{\ell+1} x 1)
    for a, w, b in zip(reversed(activations[:-1]), reversed(weights), reversed(bias)):
        # invariant: we are looking at level \ell
        # prev := dz^{\ell + 1}
        
        # d z_i^l+1 / d z_j^l = w_{ij}^l \sigma(z_j^l)(1-\sigma(z_j^l))
        # = w_{ij}^l a_j^l(1-a_j^l)
        dz = w * (a*(1-a)).T # dz_{ij} = w_{ij}^\ell a_j(1-a_j) achieved with broad casting
        #curr = (prev.T @ dz).T
        curr = dz.T @ prev
        deltas.append(curr)
        prev = curr

    assert len(deltas) == len(activations)
        
    gradients_w = [d @ a.T for d, a in zip(deltas[1:], reversed(activations[0:-1]))]
    gradients_b = [d for d in deltas[1:]]

    return gradients_w, gradients_b

def backpropagation_iteration_matrix(X, Y, activations, weights, bias):
    """
    Returns the differential qoutient of each
    edge in the neural network.
    Formally, between layers l and l+1 we compute
    dz^{l+1}_i/dz^{l}_j, since that allows us to compute
    d E_in / d z^{l}_{i}, which in turn allows us to
    compute d E_in / d w_{i,j}^l and d E_in / d b_i^l
    """
    n,d = X.shape
    assert Y.shape == (n, weights[-1].shape[0])
    assert len(activations) == len(weights)+1 == len(bias)+1
    assert all(a.shape[-1] == w.shape[1] for a, w in zip(activations[:-1], weights))
    assert all(a.shape[-1] == w.shape[0] for a, w in zip(activations[1:], weights))
    
    output = activations[-1]

    assert output.shape == Y.shape
    
    dz = -(Y - output)*output*(1-output) # matrix of shape (n x n_L)
    deltas = [dz]
    prev = dz # invariant: prev.shape = (n x n_{\ell+1})
    for a, w, b in zip(reversed(activations[:-1]), reversed(weights), reversed(bias)):
        # invariant: we are looking at level \ell
        # prev := dz^{\ell + 1}
        
        # d z_i^l+1 / d z_j^l = w_{ij}^l \sigma(z_j^l)(1-\sigma(z_j^l))
        # = w_{ij}^l a_j^l(1-a_j^l)
        dz = w[np.newaxis, :, :] * (a*(1-a))[:, np.newaxis, :]
        # dz_{kij} = w_{ij}^\ell a_{kj}(1-a_{kj}) achieved with broad casting, k refers to input number.

        # dz.shape = (n x n_{\ell+1} x n_{\ell})
        curr = np.sum(dz*prev[:,:,np.newaxis],axis=1) # uhm.. yeah.. this seems to be correct
        # curr.shape = (n,n_{\ell})
        deltas.append(curr)
        prev = curr

    assert len(deltas) == len(activations)
    deltas.reverse()

    
    gradients_w = [(d.T @ a)/n for d, a in zip(deltas[1:], activations[:-1])] # avg gradient.

    gradients_b = [np.sum(d,axis=0).reshape(-1,1)/n for d in deltas[1:]]

    assert all(g.shape == w.shape for g, w in zip(gradients_w, weights))

    return gradients_w, gradients_b

def feed_forward_iteration(x, y, weights, bias):
    d, = x.shape
    assert 1 <= len(weights) == len(bias)
    assert weights[0].shape[1] == d
    assert all(b.shape == (w.shape[0],1) for w, b in zip(weights, bias))
    assert all(w1.shape[0] == w2.shape[1] for w1, w2 in zip(weights[:-1],weights[1:]))
    """
    Returns the activation of each node in
    the neural network.
    """
    previous_activation = x.reshape(-1,1)
    activations = [previous_activation]
    for w, b in zip(weights, bias):
        z = w @ previous_activation + b
        previous_activation = sigmoid(z)
        activations.append(previous_activation)

    return activations

def feed_forward_iteration_matrix(X, Y, weights, bias):
    n,d = X.shape
    assert 1 <= len(weights) == len(bias)
    assert weights[0].shape[1] == d
    assert all(b.shape == (w.shape[0],1) for w, b in zip(weights, bias))
    assert all(w1.shape[0] == w2.shape[1] for w1, w2 in zip(weights[:-1],weights[1:]))
    assert Y.shape == (n,weights[-1].shape[0])

    
    """
    Returns the activation of each node in
    the neural network.
    """
    previous_activation = X
    activations = [previous_activation]
    for w, b in zip(weights, bias):
        z = previous_activation @ w.T + b.T
        previous_activation = sigmoid(z)
        activations.append(previous_activation)

    return activations

def regularize(gradient, weights, l=0.0):
    return [g+l*w for g, w in zip(gradient, weights)]

def backpropagation(X, Y, s, l=0.001):
    """
    We assume the neural network consists of
    n_layers = len(s) layers and layer l has s[l] nodes.
    The first layer is the input and has size
    s[0] == d. Layer i and layer i+1 form a complete
    bipartite graph. Furthermore all edges point forward.

    x and y is one input example from the data set D=(X,Y).
    """
    n,d = X.shape
    assert s[0] == d
    # initialize weights.
    # there are s[i]*s[i+1] edges between layers i and i+1
    # W[l][i][j] gives the weight on the edge from node j
    # in layer l to node i in layer l+1.
    W = [np.random.randn(s[i+1],s[i])/s[i+1] for i in range(len(s)-1)]
    bias = [np.ones((s[i+1],1)) for i in range(len(s)-1)]
    iterations_left = 1000
    eta = 1
    t0 = time.time()
    while iterations_left > 0:
        iterations_left -= 1
        #sample = np.random.choice(X.shape[0])
        #x,y = X[sample],Y[sample]
        activations = feed_forward_iteration_matrix(X, Y, W, bias)
        gradients_w,gradients_b = backpropagation_iteration_matrix(X, Y, activations, W, bias)
        gradients_w = regularize(gradients_w, W, l)

        cost = np.sum(((activations[-1]-Y)**2))/(2*n)
        ein = np.sum((activations[-1] > 0.5) != (Y > 0.5))
        t_now = time.time()
        print("[%8.4f] %5d %0.5f %5d"%(t_now-t0,iterations_left,cost,ein))
        # update weights and bias.
        W_new = [w-eta*grad_w for w, grad_w in zip(W,gradients_w)]
        W = W_new
        bias_new = [b-eta*grad_b for b, grad_b in zip(bias, gradients_b)]
        bias = bias_new
        
        
        
    return W, bias

