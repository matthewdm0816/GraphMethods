import torch
import torch.nn as nn

import torchdiffeq as tde 
from torchdiffeq import odeint_adjoint, odeint as odeint_normal

class SequentialFlow(nn.Module):
    """
    nn.Sequential container for CNFs
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, c, logpx=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.layers) - 1, -1, -1)
            else:
                inds = range(len(self.layers))

        if logpx is None:
            for i in inds:
                x = self.layers[i](x, c, logpx=logpx, integration_times=integration_times, reverse=reverse)
        else:
            for i in inds:
                x, logpx = self.layers[i](x, c, logpx=logpx, integration_times=integration_times, reverse=reverse)

class CNF(nn.Module):
    """
    Continuous Normalizing Flow
    integrate df through time
    """
    def __init__(self, df, conditional=True, T=1., train_T=False, 
        regularization=None, solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super().__init__()
        self.train_T, self.T, self.df = train_T, T, df
        if train_T:
            self.register_parameter("sqrt_time", nn.Parameter(torch.tensor(T).sqrt()))
        
        # ! Hmmmmmmmm
        if regularization is not None and len(regularization) != 0:
            raise NotImplementedError("Regularization Not Implemented Yet!")

        self.solver, self.atol, self.rtol, self.use_adjoint, self.conditional = \
            solver, atol, rtol, use_adjoint, conditional

        self.solver_args = {}

    def forward(self, x, c=None, logpx=None, integration_times=None, reverse=False):
        if logpx is None:
            # initial logpx ~ $-\infty$
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx
        
        # set if conditional/parametric(to c)
        if self.conditional:
            assert c is not None
            states = (x, _logpx, c)
            atol = [self.atol] * 3
            rtol = [self.rtol] * 3
        else:
            states = (x, _logpx)
            atol = [self.atol] * 2
            rtol = [self.rtol] * 2

        # set integration time
        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack([
                    torch.tensor(0.).to(x), self.sqrt_time ** 2
                ]).to(x)
            else:
                integration_times = torch.tensor([
                    0, self.T
                ], requires_grad=False).to(x)

        if reverse:
            _flip(integration_times, 0)

        self.df.before_odeint()
        _odeint = odeint_adjoint if self.use_adjoint else odeint_normal

        if self.training:
            intf = _odeint(
                self.df, states, integration_times,
                atol=atol, rtol=rtol,
                method=self.solver,
                options=self.solver_args
            )
        else:
            intf = _odeint(
                self.df, states, integration_times,
                atol=self.atol, rtol=self.rtol,
                method=self.solver,
                options=self.solver_args
            )

        if len(integration_times) == 2: # ? a must one ?
            z_t, logpz_t = tuple(
                s[1] for s in intf
            )
        
        if logpx is not None:
            return z_t, logpz_t
        else:
            return intf
    
    def num_evals(self):
        return self.df._num_evals.item()


def _flip(x, dim=0):
    indices = [slice(None)] * 3
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long).to(x)
    return x[tuple(indices)]

