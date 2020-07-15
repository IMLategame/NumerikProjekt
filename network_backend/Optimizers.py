from network_backend.Modules import ModuleI
import numpy as np


class OptimizerI:
    # call super(subclass, self).__init__(net) in subclasses
    def __init__(self, net):
        assert isinstance(net, ModuleI)
        self.net = net

    # take one step towards optimum based on previous backpropagation
    def take_step(self):
        raise NotImplementedError()

"""
    The classic Optimizer is also the simplest: Vanilla stochastic gradient descent
"""
class SGD(OptimizerI):
    def __init__(self, net, rate):
        super().__init__(net)
        self.rate = rate

    def take_step(self):
        gradients = self.net.getGradients()
        delta = [self.rate * grad for grad in gradients]
        self.net.update(delta)

"""
    This optimizer is more advanced. It uses the first two moments.
    If we went into some direction for a few steps already,
    might start to take bigger steps into said direction.
    
    alpha = step size
    beta_1 = decay rate for first moment (in [0,1])
    beta_2 = decay rate for second moment (in [0,1])
    eps: to not divide by zero
    
    For further explanation see https://arxiv.org/pdf/1412.6980.pdf
"""
class Adam(OptimizerI):
    def __init__(self, net, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 10e-8):
        super(Adam, self).__init__(net)
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        # don't initialize the first moment m and second moment v, yet
        self.m = None
        self.v = None

    def take_step(self):
        # take care: when dealing with np arrays '*' and '/' are pointwise operations. '@' is the matrix multiplication
        gradients = self.net.getGradients()
        if self.m is None:
            # initialize with zeros
            self.m = [np.zeros(grad.shape) for grad in gradients]
            self.v = [np.zeros(grad.shape) for grad in gradients]
        self.m = [self.beta_1 * m_i + (1 - self.beta_1) * grad for m_i, grad in zip(self.m, gradients)]
        gradients_squared = [grad*grad for grad in gradients]
        self.v = [self.beta_2 * v_i + (1 - self.beta_2) * grad_sq for v_i, grad_sq in zip(self.v, gradients_squared)]

        m_bias_corrected = [1/(1 - self.beta_1)*m_i for m_i in self.m]
        v_bias_corrected = [1/(1 - self.beta_2)*v_i for v_i in self.v]

        delta = [self.alpha * m_i / (np.sqrt(v_i)+self.eps) for m_i, v_i in zip(m_bias_corrected, v_bias_corrected)]

        self.net.update(delta)