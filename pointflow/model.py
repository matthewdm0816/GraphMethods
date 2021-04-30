import torch
import torch.nn as nn
from odenet import ODEFunction, ODENet 
from cnf import CNF, SequentialFlow
from utils import dictobj

def get_accumulater(crit, counter):
    """
    build a universal accumulater for nn.Module.apply(fn)
    """
    class Accumulater():
        def __init__(self):
            self.total = 0

        def __call__(self, module):
            if crit(module):
                self.total += counter(module)
        
    return Accumulater()

def count_model(model: nn.Module, accumulater):
    model.apply(accumulater)
    return accumulater.total

count_eval_times = lambda model: \
    count_model(model, get_accumulater(lambda module: isinstance(module, CNF), lambda module: module._eval_times))
count_total_time = lambda model: \
    count_model(model, get_accumulater(lambda module: isinstance(module, CNF), lambda module: module.sqrt_time ** 2.))

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(fin, fhidden, fcontext, num_blocks, conditional, **args):
    args = dictobj(args)
    def build_cnf():
        diffeq = ODENet(
            fhidden=fhidden,
            fcontext=fcontext,
            in_shape=(fin,),
            layer_type=args.layer_type
        )
        odefunc = ODEFunction(
            diffeq=diffeq
        )
        cnf = CNF(
            df=odefunc,
            T=args.T, train_T=args.train_T, conditional=conditional, 
            solver=args.solver, use_adjoint=args.use_adjoint, 
            atol=args.atol, rtol=args.rtol
        )
        return cnf

    layers = [build_cnf() for _ in range(num_blocks)]
    if args.bn:
        # TODO: Implement Moving Batch Norm
        raise NotImplementedError("Moving Batch Normalization Layer Not Implemented!")
    model = SequentialFlow(layers)
    return model

def get_point_cnf(args):
    otherargs = args.copy()
    del otherargs['fin'], otherargs['num_blocks']

    model = build_model(fin=args.fin, fhidden=args.dims, fcontext=args.fz, 
        num_blocks=args.num_blocks, conditional=True, **otherargs).cuda()
    print("Parameters in Point-CNF %d." % count_parameters(model))
    return model

def get_latent_cnf(args):
    """
    latent CNF: use non-parametrized f(z(t))
    """
    otherargs = args.copy()
    del otherargs['fin'], otherargs['num_blocks']

    model = build_model(fin=args.fz, fhidden=args.latent_dims, fcontext=0, 
        num_blocks=args.latent_num_blocks, conditional=False, **otherargs).cuda() # ? why not args.fin ?
    print("Parameters in Latent-CNF %d." % count_parameters(model))
    return model