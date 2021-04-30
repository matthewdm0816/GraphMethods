import torch
import torch.nn as nn
import torch.nn.functional as F

import diffop

class Swish(nn.Module):
    """
    Swish activation: $x' = x * \\exp(\\beta x)$
    """
    def __init__(self, beta):
        super().__init__()

    def forward(self, x):
        return x * F.tanh(F.softplus(x))

class Mish(nn.Module):
    """
    Mish activation: $x' = x * \\exp(\\beta x)$
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)

class Square(nn.Module):
    """
    Square activation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x

def div_approx(f, y, e=None):
    """
    Hutchinson Estimator 
    $$
    \\operatorname{Tr}(\\frac{\\partial f}{\\partial z(t)}) 
        = \mathbb{E}_{p(\epsilon)}\left[\epsilon^{T} \frac{\partial f}{\partial z(t)} \epsilon\right]
    $$
    simply speaking, Tr[A] ~= E_{p_e}[e^T A e]
    in this work we'll set noise e ~ N(0, I) to simplify computations
    for theoretical issue, refer to the article of FFJORD
    """
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)
    
    if not e_dzdx_e.requires_grad:
        for _ in range(10): # approx for 10 times in Hutchinson Estimator, accumulate grad!
            e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
            e_dzdx_e = e_dzdx.mul(e)
    
    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad
    return approx_tr_dzdx


class ODENet(nn.Module):
    """
    combine diffops into f(z(t))
    """
    def __init__(self, fhidden, in_shape, fcontext, layer_type=diffop.CoScaleLinear, nonlinearity=nn.Softplus):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        self.activations = nn.ModuleList([])
        hidden_shape = in_shape[0]

        for f_out in (fhidden + (in_shape[0],)):
            _layer = layer_type(fin=hidden_shape, fout=f_out, fcontext=fcontext, **{})
            self.layers.append(_layer)
            self.activations.append(nonlinearity())

            hidden_shape = f_out

    def forward(self, context, y):
        dx = y
        for i, layer in enumerate(self.layers):
            dx = layer(context, dx)
            if i != len(self.layers) - 1:
                dx = self.activations[i](dx)
            # print(dx.shape)
        return dx

class ODEFunction(nn.Module):
    """
    self.diffeq => the ODENet f(z(t))
    this class computes derivatives df/dz(t)
    """
    def __init__(self, diffeq: ODENet):
        super().__init__()
        self.diffeq = diffeq
        self.div_fn = div_approx
        self.register_buffer('_num_evals', torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        # print(len(states))
        y = states[0]
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y) 
            # expanded to many Ts, y.size(0) => batch_size
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        if self._e is None: # if not specified a noise, fix one(for each epoch)
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True): # ? why to set enabled, isn't it enabled already
            if len(states) == 3: # conditional CNF
                c = states[-1]
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=1)
                dy = self.diffeq(tc, y)
                div = self.div_fn(dy, y, e=self._e).unsqueeze(-1)
                return dy, -div, torch.zeros_like(c).requires_grad_(True) # ? why last return value(zero tensor)

            elif len(states) == 2: # unconditional CNF
                dy = self.diffeq(t, y)
                div = self.div_fn(dy, y, e=self._e).view(-1, 1)
                return dy, -div
                
            else:
                raise Exception("states length must be 2 or 3")



        
