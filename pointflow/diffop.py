import torch
import torch.nn as nn
import torch.nn.init as init

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

"""
Definitions of argumented derivatives
x: input of $x(t)$
c: context, the concat of argument $\boldsymbol{z}$ and time $t$
"""
class DirectLinear(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.net = nn.Linear(fin, fout)

    def forward(self, c, x): 
        return self.net(x)

class CoLinear(nn.Module):
    def __init__(self, fin, fcontext, fout):
        super().__init__()
        self.net = nn.Linear(fin + fcontext + 1, fout)

    def forward(self, c, x):
        if x.dim() == 3:
            c = c.unsqueeze(1).expand(-1, x.size(1), -1)
        x_c = torch.cat(x, c, dim=2)
        return self.net(x_c)

class SeperateCoLinear(nn.Module):
    def __init__(self, fin, fcontext, fout):
        super().__init__()
        self.net = nn.Linear(fin, fout)
        self.bias = nn.Linear(fcontext + 1, fout)

    def forward(self, c, x):
        bias = self.bias(c)
        if x.dim() == 3:
            bias = bias.unsqueeze(1)
        return self.net(x) + bias

class ScaleLinear(nn.Module):
    """
    activation => ``None``/``torch.nn.Sigmoid``
    """
    def __init__(self, fin, fcontext, fout, activation=nn.Sigmoid):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(fcontext + 1, fout, bias=False),
            activation() if activation is not None else nn.Identity()
        )
        self.net = nn.Linear(fin, fout)

    def forward(self, c, x):
        scale = self.gate(c)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
        return self.net(x) * scale

class ScaleDirectLinear(ScaleLinear):
    """
    activation => ``None``/``torch.nn.Sigmoid``
    """
    def __init__(self, fin, fcontext, fout):
        super().__init__(fin, fcontext, fout, activation=None)
        
class CoScaleLinear(nn.Module):
    """
    activation => ``None``/``torch.nn.Sigmoid``
    """
    def __init__(self, fin, fcontext, fout, activation=nn.Sigmoid):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(fcontext + 1, fout, bias=False),
            activation() if activation is not None else nn.Identity()
        )
        self.net = nn.Linear(fin, fout)
        self.bias = nn.Linear(fcontext + 1, fout)
        print("Built a %s layer with fin=%d, fout=%d, fcontext=%d" %
                (self.__class__.__name__, fin, fout, fcontext))

    def forward(self, c, x):
        # print("Input shapes: ", c.shape, x.shape)
        scale = self.gate(c)
        bias = self.bias(c)
        lin = self.net(x)
        # print("Intermediate shapes:", scale.shape, bias.shape, lin.shape)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return lin * scale + bias
    
class CoScaleDirectLinear(CoScaleLinear):
    """
    activation => ``None``/``torch.nn.Sigmoid``
    """
    def __init__(self, fin, fcontext, fout):
        super().__init__(fin, fcontext, fout, activation=None)