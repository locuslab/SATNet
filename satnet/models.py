import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim

import satnet._cpp
if torch.cuda.is_available(): import satnet._cuda


def get_k(n):
    return int((2*n)**0.5+3)//4*4

class MixingFunc(Function):
    '''Apply the Mixing method to the input probabilities.

    Args: see SATNet.

    Impl Note: 
        The SATNet is a wrapper for the MixingFunc,
        handling the initialization and the wrapping of auxiliary variables.
    '''
    @staticmethod
    def forward(ctx, S, z, is_input, max_iter, eps, prox_lam):
        B, n, m, k = z.size(0), S.size(0), S.size(1), 32 #get_k(S.size(0))
        ctx.prox_lam = prox_lam
        
        device = 'cuda' if S.is_cuda else 'cpu'
        ctx.g, ctx.gnrm = torch.zeros(B,k, device=device), torch.zeros(B,n, device=device)
        ctx.index = torch.zeros(B,n, dtype=torch.int, device=device)
        ctx.is_input = torch.zeros(B,n, dtype=torch.int, device=device)
        ctx.V, ctx.W = torch.zeros(B,n,k, device=device).normal_(), torch.zeros(B,k,m, device=device)
        ctx.z = torch.zeros(B,n, device=device)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=device)

        ctx.S = torch.zeros(n,m, device=device)
        ctx.Snrms = torch.zeros(n, device=device)

        ctx.z[:] = z.data
        ctx.S[:] = S.data
        ctx.is_input[:] = is_input.data

        perm = torch.randperm(n-1, dtype=torch.int, device=device)

        satnet_impl = satnet._cuda if S.is_cuda else satnet._cpp
        satnet_impl.init(perm, is_input, ctx.index, ctx.z, ctx.V)

        for b in range(B):
            ctx.W[b] = ctx.V[b].t().mm(ctx.S)
        ctx.Snrms[:] = S.norm(dim=1)**2
        
        satnet_impl.forward(max_iter, eps, 
                ctx.index, ctx.niter, ctx.S, ctx.z, 
                ctx.V, ctx.W, ctx.gnrm, ctx.Snrms, ctx.g)

        return ctx.z.clone()
    
    @staticmethod
    def backward(ctx, dz):
        B, n, m, k = dz.size(0), ctx.S.size(0), ctx.S.size(1), 32 #get_k(ctx.S.size(0))

        device = 'cuda' if ctx.S.is_cuda else 'cpu'
        ctx.dS = torch.zeros(B,n,m, device=device)
        ctx.U, ctx.Phi = torch.zeros(B,n,k, device=device), torch.zeros(B,k,m, device=device)
        ctx.dz = torch.zeros(B,n, device=device)

        ctx.dz[:] = dz.data

        satnet_impl = satnet._cuda if ctx.S.is_cuda else satnet._cpp
        satnet_impl.backward(ctx.prox_lam, 
                ctx.is_input, ctx.index, ctx.niter, ctx.S, ctx.dS, ctx.z, ctx.dz,
                ctx.V, ctx.U, ctx.W, ctx.Phi, ctx.gnrm, ctx.Snrms, ctx.g)

        ctx.dS = ctx.dS.sum(dim=0)

        return ctx.dS, ctx.dz, None, None, None, None

def insert_constants(x, pre, n_pre, app, n_app):
    ''' prepend and append torch tensors
    '''
    one  = x.new(x.size()[0],1).fill_(1)
    seq = []
    if n_pre != 0:
        seq.append((pre*one).expand(-1, n_pre))
    seq.append(x)
    if n_app != 0:
        seq.append((app*one).expand(-1, n_app))
    r = torch.cat(seq, dim=1)
    r.requires_grad = False
    return r

class SATNet(nn.Module):
    '''Apply a SATNet layer to complete the input probabilities.

    Args:
        n: Number of input variables.
        m: Rank of the clause matrix.
        aux: Number of auxiliary variables.

        max_iter: Maximum number of iterations for solving
            the inner optimization problem.
            Default: 40
        eps: The stopping threshold for the inner optimizaiton problem.
            The inner Mixing method will stop when the function decrease
            is less then eps times the initial function decrease.
            Default: 1e-4
        prox_lam: The diagonal increment in the backward linear system
            to make the backward pass more stable.
            Default: 1e-2
        weight_normalize: Set true to perform normlization for init weights.
            Default: True

    Inputs: (z, is_input)
        **z** of shape `(batch, n)`: 
            Float tensor containing the probabilities (must be in [0,1]).
        **is_input** of shape `(batch, n)`: 
            Int tensor indicating which **z** is a input.

    Outputs: z
        **z** of shape `(batch, n)`:
            The prediction probabiolities.

    Attributes: S
        **S** of shape `(n, m)`:
            The learnable clauses matrix containing `m` clauses 
            for the `n` variables.

    Examples:
        >>> sat = satnet.SATNet(3, 4, aux=5)
        >>> z = torch.randn(2, 3)
        >>> is_input = torch.IntTensor([[1, 1, 0], [1,0,1]])
        >>> pred = sat(z, is_input)
    '''

    def __init__(self, n, m, aux=0, max_iter=40, eps=1e-4, prox_lam=1e-2, weight_normalize=True):
        super(SATNet, self).__init__()

        S_t = torch.FloatTensor(n+1+aux, m)    # n+1 for truth vector
        S_t = S_t.normal_() 
        if weight_normalize: S_t = S_t * ((.5/(n+1+aux+m))**0.5)

        self.S = nn.Parameter(S_t)
        self.aux = aux 
        self.max_iter, self.eps, self.prox_lam = max_iter, eps, prox_lam

    def forward(self, z, is_input):
        B = z.size(0)
        device = 'cuda' if self.S.is_cuda else 'cpu'
        m = self.S.shape[1]
        if device == 'cpu' and m%4 != 0:
            raise ValueError('m is required to be a multiple of 4 on CPU for SSE acceleration. Now '+str(m))
        is_input = insert_constants(is_input.data, 1, 1, 0, self.aux)
        z = torch.cat([torch.ones(z.size(0),1,device=device), z, torch.zeros(z.size(0),self.aux,device=device)],dim=1)

        z = MixingFunc.apply(self.S, z, is_input, self.max_iter, self.eps, self.prox_lam)

        return z[:,1:self.S.size(0)-self.aux]
