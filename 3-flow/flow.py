import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchdiffeq import odeint_adjoint as odeint
from itertools import permutations


class odeModule(nn.Module):
    # for torchdiffeq
    def __init__(self, net, device='cpu', name=None, checkpoint=False):
        super(odeModule, self).__init__()
        self.device = device
        if name is None:
            self.name = 'odeModule'
        else:
            self.name = name
        self.net = net 
        self.checkpoint = checkpoint
    
    def forward(self, t, states):
        # states=(x, logp)
        x = states[0]
        logp = states[1]
        if self.checkpoint:
            return self.checkpoint(self.net.grad, x), -self.checkpoint(self.net.laplacian, x)
        else:
            return self.net.grad(x), -self.net.laplacian(x)



class MongeAmpereFlow(nn.Module):
    '''
    Monge-Ampere Flow for generative modeling 
    https://arxiv.org/abs/1809.10188
    dx/dt = grad u(x)
    dlnp(x)/dt = -laplacian u(x) 
    '''
    def __init__(self, net, epsilon, Nsteps, device='cpu', name=None, checkpoint=False):
        super(MongeAmpereFlow, self).__init__()
        self.device = device
        if name is None:
            self.name = 'MongeAmpereFlow'
        else:
            self.name = name
        self.net = net 
        self.dim = net.dim
        self.epsilon = epsilon 
        self.Nsteps = Nsteps
        self.checkpoint = checkpoint
        self.odeModule = odeModule(net, device, checkpoint=checkpoint)

    def integrate(self, x, logp, sign=1, epsilon=None, Nsteps=None):
        #default values
        if epsilon is None:
            epsilon = self.epsilon 
        if Nsteps is None:
            Nsteps = self.Nsteps

        # integrate ODE for x and logp(x)
        # def ode(x):
        #     if self.checkpoint:
        #         return sign*epsilon*checkpoint(self.net.grad, x), -sign*epsilon*checkpoint(self.net.laplacian, x)
        #     else:
        #         return sign*epsilon*self.net.grad(x), -sign*epsilon*self.net.laplacian(x)

        # rk4 Runge-Kutta-4
        # for step in range(Nsteps):
        #     k1_x, k1_logp = ode(x)
        #     k2_x, k2_logp = ode(x+k1_x/2)
        #     k3_x, k3_logp = ode(x+k2_x/2)
        #     k4_x, k4_logp = ode(x+k3_x)

        #     x = x + (k1_x/6.+k2_x/3. + k3_x/3. +k4_x/6.) 
        #     logp = logp + (k1_logp/6. + k2_logp/3. + k3_logp/3. + k4_logp/6.)

        # Euler
        # dx, dlogp = ode(x)
        # x = x + dx
        # logp = logp + dlogp

        # return x, logp

        #torchdiffeq
        x_t, logp_t = odeint(
            self.odeModule,
            (x, logp),
            torch.tensor([0, sign*Nsteps], requires_grad=True, dtype=torch.float32).to(self.device),
            atol=1e-7,
            rtol=1e-7,
            method='dopri5',
            )
                
        return x_t[-1], logp_t[-1]

    def sample(self, batch_size):
        #initial value from Gaussian
        x = torch.randn(batch_size, self.dim, device=self.device, requires_grad=True)
        logp = -0.5 * x.pow(2).add(math.log(2 * math.pi)).sum(1) 
        return self.integrate(x, logp, sign=1)

    def nll(self, x):
        '''
        integrate backwards, thus it returns logp(0) - logp(T)
        '''
        logp = torch.zeros(x.shape[0], device=x.device) 
        x, logp = self.integrate(x, logp, sign=-1)
        return logp + 0.5 * x.pow(2).add(math.log(2 * math.pi)).sum(1)

    def check_reversibility(self, x, logp):
        z, logp_z = self.integrate(x, logp, sign=-1)
        x_back, logp_back = self.integrate(z, logp_z, sign=1)

        x_error = ((x-x_back).abs().sum()) # check reversibility 
        logp_error = ((logp- logp_back).abs().sum())
        return x_error, logp_error
    
    def check_permSym(self, x, logp):
        z, logp_z = self.integrate(x, logp, sign=-1)
        logp_error = []
        for permute in permutations(list(range(int(self.dim/2)))):
            xIndex = torch.LongTensor(permute).to(z.device)*2
            yIndex = xIndex + 1
            index = torch.vstack((xIndex, yIndex)).transpose(1, 0).flatten()
            zPerm = torch.index_select(z, dim=1, index=index)
            logpPerm = self.integrate(zPerm, logp_z, sign=1)[1]
            logp_error.append((logpPerm-logp).abs().sum())
        return logp_error


if __name__=='__main__':
    from net import Simple_MLP
    numElectron = 4
    net = Simple_MLP(dim=2*numElectron, hidden_size = 32, permSym=True)
    model = MongeAmpereFlow(net, epsilon=0.1, Nsteps=100)
    x, logp = model.sample(10)
    print ('Checking Reversibility, x: %f, logp: %f' % (model.check_reversibility(x, logp)))
    # print ('Checking Permutation Symmetry: ', list(map(float, model.check_permSym(x, logp))))
