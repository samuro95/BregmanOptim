import torch
import torch.nn as nn
from deepinv.optim.utils import check_conv, gradient_descent

class FixedPointOptim(nn.Module):
    '''
    Fixed Point Optimization algorithms for minimizing the sum of two functions \lambda*f + g where f is a data-fidelity term that will me modeled by an instance of physics
    and g is a regularizer either explicit or implicitly given by either its prox or its gradient. 
    By default, the algorithms starts with a step on f and finishes with step on g. 

    :param data_fidelity: data_fidelity instance modeling the data-fidelity term.   
    :param lamb: Regularization parameter.
    :param g: Regularizing potential. 
    :param prox_g: Proximal operator of the regularizing potential. x,it -> prox_g(x,it)
    :param grad_g: Gradient of the regularizing potential. x,it -> grad_g(x,it)
    :param max_iter: Number of iterations.
    :param step_size: Step size of the algorithm. List or int. If list, the length of the list must be equal to max_iter.
    :param theta: Relacation parameter of the ADMM/DRS/PD algorithms.
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param crit_conv: Mimimum relative change in the solution to stop the algorithm.
    :param unroll: If True, the algorithm is unrolled in time.
    :param verbose: prints progress of the algorithm.
    '''

    def __init__(self, data_fidelity='L2', lamb=1., device='cpu', g = None, prox_g = None,
                 grad_g = None, max_iter=10, stepsize=1., theta=1., g_first = False, crit_conv=None, unroll=False,
                 weight_tied=True, verbose=False, stepsize_inter = 1., max_iter_inter=50, tol_inter=1e-3, init=None) :
        super().__init__()

        self.data_fidelity = data_fidelity
        self.lamb = lamb
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_first = g_first
        self.unroll = unroll
        self.weight_tied = weight_tied
        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.device = device
        self.has_converged = False
        self.init = init

        if g is not None and isinstance(g, nn.Module) :
            def grad_g(self,x,it):
                torch.set_grad_enabled(True)
                return torch.autograd.grad(g(x), x, create_graph=True, only_inputs=True)[0]
            def prox_g(self,x,it) :
                grad = lambda  y : grad_g(y,it) + (1/2)*(y-x)
                return gradient_descent(grad, x, stepsize_inter, max_iter=max_iter_inter, tol=tol_inter)

        if isinstance(stepsize, float):
            stepsize = [stepsize] * max_iter
        elif isinstance(stepsize, list):
            assert len(stepsize) == max_iter
            stepsize = stepsize
        else:
            raise ValueError('stepsize must be either int/float or a list of length max_iter') 
        
        if self.unroll : 
            self.register_parameter(name='stepsize',
                                param=torch.nn.Parameter(torch.tensor(stepsize, device=self.device),
                                requires_grad=True))
        else:
            self.stepsize = stepsize
        
        if isinstance(theta, float):
            self.theta = [theta] * max_iter
        elif isinstance(theta, list):
            assert len(theta) == max_iter
            self.theta = theta
        else:
            raise ValueError('theta must be either int/float or a list of length max_iter') 

        def FP_operator(self, x, y, physics, it):
            pass

        def forward(self, y, physics):
            x = self.init if self.init is not None else y
            for it in range(self.max_iter):
                x_prev = x
                x = FP_operator(x,y,physics)
                if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose):
                    break
            return x 

class GD(FixedPointOptim):

    def __init__(**kwards):
        super().__init__(**kwards)
    
    def FP_operator(self, x, y, physics, it):
        return x - self.stepsize[it]*(self.lamb*self.data_fidelity.grad(x, y, physics) + self.grad_g(x,it))


class HQS(FixedPointOptim):

    def __init__(**kwards):
        super().__init__(**kwards)
    
    def FP_operator(self, x, y, physics, it):
        if not self.g_first : 
            z = self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize[it])
            x = self.prox_g(z, it)
        else :
            z = self.prox_g(z, it)
            x = self.data_fidelity.prox(z, y, physics, self.lamb*self.stepsize[it])
        return x

class PGD(FixedPointOptim):

    def __init__(**kwards):
        super().__init__(**kwards)
    
    def FP_operator(self, x, y, physics, it):
        if not self.g_first : # prox on g and grad on f
            z = x - self.stepsize[it]*self.lamb*self.data_fidelity.grad(x, y, physics)
            x = self.prox_g(z, it)
        else :  # prox on f and grad on g
            z = x - self.stepsize[it]*self.grad_g(x,it)
            x = self.data_fidelity.prox(z, y, physics, self.lamb*self.stepsize[it])
        return x

class DRS(FixedPointOptim):

    def __init__(**kwards):
        super().__init__(**kwards)
    
    def FP_operator(self, x, y, physics, it):
        if not self.g_first :
            Rprox_f = 2*self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize[it])-x
            Rprox_g = 2*self.prox_g(Rprox_f, it)-Rprox_f
            x = (1/2)*(x + Rprox_g)
        else :
            Rprox_g = 2*self.prox_g(x, it)-x
            Rprox_f = 2*self.data_fidelity.prox(Rprox_g, y, physics, self.lamb*self.stepsize[it])-Rprox_g
            x = (1/2)*(x + Rprox_g)
        return x

class ADMM(FixedPointOptim):

    def __init__(**kwards):
        super().__init__(**kwards)

    def FP_operator(self, x, y, physics, it):
        # TODO : same as DRS ???
    

        

# def ADMM(self, y, physics, init=None):
#         '''
#         Alternating Direction Method of Multipliers (ADMM)
#         :param y: Degraded image.
#         :param physics: Physics instance modeling the degradation.
#         :param init: Initialization of the algorithm. If None, the algorithm starts from y.
#         '''
#         if init is None:
#             w = y
#         else:
#             w = init
#         x = torch.zeros_like(w)
#         for it in range(self.max_iter):
#             x_prev = x
#             if not self.g_first :
#                 z = self.data_fidelity.prox(w-x, y, physics, self.lamb*self.stepsize[it])
#                 w = self.prox_g(z+x_prev, it)
#             else :
#                 z = self.prox_g(w-x, it)
#                 w = self.data_fidelity.prox(z+x_prev, y, physics, self.lamb*self.stepsize[it])
#             x = x_prev + self.theta[it]*(z - w)
#             if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose) :
#                 break
#         return w
