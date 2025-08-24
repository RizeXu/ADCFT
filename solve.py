import torch
from Operator import Operator
from EigenSolver import EigenSolver
import numpy as np
import scipy

def solve(op : Operator,
          B2 : torch.Tensor = torch.tensor([]),
          B4 : torch.Tensor = torch.tensor([]),
          B6 : torch.Tensor = torch.tensor([])):
    r"""
    solve the systems with the steven operators
    Args:
        op (Operator): the steven operator
        B2 (torch.Tensor): the coefficient for O[2,m]
        B4 (torch.Tensor): the coefficient for O[4,m]
        B6 (torch.Tensor): the coefficient for O[6,m]
    Returns:
        w (torch.Tensor): the eigen energy
        v (torch.Tensor): the eigen states
    """
    H = torch.zeros((op.N, op.N), dtype=torch.complex64)

    if isinstance(B2, np.ndarray):
        B2 = torch.tensor(B2, dtype=torch.complex64)
    if isinstance(B4, np.ndarray):
        B4 = torch.tensor(B4, dtype=torch.complex64)
    if isinstance(B6, np.ndarray):
        B6 = torch.tensor(B6, dtype=torch.complex64)

    if B2.shape[0] != 0:
        H += torch.einsum('i,ijk->jk', B2.to(torch.complex64), op.O2)
    if B4.shape[0] != 0:
        H += torch.einsum('i,ijk->jk', B4.to(torch.complex64), op.O4)
    if B6.shape[0] != 0:
        H += torch.einsum('i,ijk->jk', B6.to(torch.complex64), op.O6)

    # assert torch.dist(H.T.conj(), H) < 1.0e-6

    w, u = EigenSolver.apply(H)

    return w, u

def measure(op : Operator, kT : torch.Tensor,
            B2 : torch.Tensor = torch.tensor([]),
            B4 : torch.Tensor = torch.tensor([]),
            B6 : torch.Tensor = torch.tensor([])):
    r"""
    measure the energy and specific heat with steven operators
    Args:
        op (Operator): the steven operator
        kT (torch.Tensor): the energy
        B2 (torch.Tensor): the coefficient for O[2,m]
        B4 (torch.Tensor): the coefficient for O[4,m]
        B6 (torch.Tensor): the coefficient for O[6,m]
    Returns:
        u (torch.Tensor): the energy
        c (torch.Tensor): the specific heat
    """

    w, _ = solve(op, B2, B4, B6)
    β = (1 / kT).requires_grad_(True)

    lnz = torch.logsumexp(-β.unsqueeze(-1) * w, dim=1)
    dlnz, = torch.autograd.grad(lnz, β, grad_outputs=torch.ones(β.shape[0]),create_graph=True)
    d2lnz, = torch.autograd.grad(dlnz, β, grad_outputs=torch.ones(β.shape[0]),create_graph=True)

    u = -dlnz
    c = d2lnz * β * β

    return u, c

def fun_loss(a: torch.Tensor, kT: torch.Tensor, cexp: torch.Tensor, op : Operator, B_func):
    r"""
    the loss of the reference specific heat

    Args:
        a (torch.Tensor): the parameters tensor
        kT (torch.Tensor): the temperature
        cexp (torch.Tensor): the heat specific
        op (Operator): the steven operators
        B_func (function): the function map a->B, where B is coefficient for steven operators

    Returns:
        loss (torch.Tensor): the loss.
        dloss (torch.Tensor): the derivative of the loss.

    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    else:
        a = a.clone().detach().requires_grad_(True)
    B2, B4, B6 = B_func(a)

    _, c = measure(op, kT, B2, B4, B6)

    loss = torch.sum((c - cexp) ** 2)
    dloss, = torch.autograd.grad(loss, a)
    return loss.detach().numpy(), dloss.detach().numpy()

def fit_c(a0, kT, cexp, op : Operator, build_B, method='L-BFGS-B', bounds=None):
    r"""
    fit the CFT parameters
    Args:
        a0: initial parameters
        kT: temperature
        cexp: reference specific heat
        op (Operator): steven operators
        method: method for minimization
        bounds: bounds for minimization
    Returns:
        res
    """
    # TODO : estimate the bounds with the real material
    if bounds is None:
        bounds = list(zip(-np.ones(a0.shape[0]), np.ones(a0.shape[0])))

    res = scipy.optimize.minimize(fun_loss,
                                  a0,
                                  args=(kT, cexp, op, build_B),
                                  method=method,
                                  bounds=bounds,
                                  jac=True, tol=1e-20)
    return res
