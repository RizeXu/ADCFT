import torch
from Operator import Operator
from EigenSolver import EigenSolver
import numpy as np
import scipy
from typing import Callable, Tuple
from zipdata import MeaData
def build_fieldx(B):
    """
    The function to build H
    default for z-axis
    Args:
        B: the magnetic field

    Returns:
        H: g μB B

    """

    zero = torch.tensor([0.0])
    g = torch.tensor([1.0, 1.0, 1.0])
    μB = 1.0

    H = torch.cat([zero, zero, B])
    H = μB * g * H
    return H

def solve(op : Operator,
          B1 : torch.Tensor = torch.tensor([]),
          B2 : torch.Tensor = torch.tensor([]),
          B4 : torch.Tensor = torch.tensor([]),
          B6 : torch.Tensor = torch.tensor([])):
    r"""
    solve the systems with the steven operators
    Args:
        op (Operator): the steven operator
        B1 (torch.Tensor): the coefficient for O[1, m], also the magnetic field, g μB B
        B2 (torch.Tensor): the coefficient for O[2,m]
        B4 (torch.Tensor): the coefficient for O[4,m]
        B6 (torch.Tensor): the coefficient for O[6,m]
    Returns:
        w (torch.Tensor): the eigen energy
        v (torch.Tensor): the eigen states
    """
    Ham = torch.zeros((op.N, op.N), dtype=torch.complex64)

    if isinstance(B1, np.ndarray):
        B1 = torch.tensor(B1, dtype=torch.complex64)
    if isinstance(B2, np.ndarray):
        B2 = torch.tensor(B2, dtype=torch.complex64)
    if isinstance(B4, np.ndarray):
        B4 = torch.tensor(B4, dtype=torch.complex64)
    if isinstance(B6, np.ndarray):
        B6 = torch.tensor(B6, dtype=torch.complex64)

    if B1.numel() != 0:
        Ham0 = torch.einsum('i,ijk->jk', B1.to(torch.complex64), op.O1)
        Ham = Ham + Ham0
        # Ham += torch.einsum('i,ijk->jk', B1.to(torch.complex64), op.O1)
    if B2.numel() != 0:
        Ham += torch.einsum('i,ijk->jk', B2.to(torch.complex64), op.O2)
    if B4.numel() != 0:
        Ham += torch.einsum('i,ijk->jk', B4.to(torch.complex64), op.O4)
    if B6.numel() != 0:
        Ham += torch.einsum('i,ijk->jk', B6.to(torch.complex64), op.O6)

    Ham = 0.5 * (Ham + Ham.T.conj())

    # assert torch.dist(H.T.conj(), H) < 1.0e-6

    w, u = EigenSolver.apply(Ham)
    # w, u = torch.linalg.eigh(Ham)

    return w, u
def measure_uc(op : Operator, kT : torch.Tensor,
               B0 : torch.Tensor = torch.tensor([]),
               field_func:Callable[[torch.Tensor], torch.Tensor]=build_fieldx,
               B2 : torch.Tensor = torch.tensor([]),
               B4 : torch.Tensor = torch.tensor([]),
               B6 : torch.Tensor = torch.tensor([])):
    r"""
    measure the energy and specific heat with steven operators
    Args:
        op (Operator): the steven operator
        kT (torch.Tensor): the temperature
        B0 (torch.Tensor): the magnetic field
        field_func (Callable[[torch.Tensor], torch.Tensor]): the field function to map B0 -> g μB B0
        B2 (torch.Tensor): the coefficient for O[2,m]
        B4 (torch.Tensor): the coefficient for O[4,m]
        B6 (torch.Tensor): the coefficient for O[6,m]
    Returns:
        u (torch.Tensor): the energy
        c (torch.Tensor): the specific heat
    """
    B1 = field_func(B0)
    w, _ = solve(op, B1, B2, B4, B6)
    β = (1 / kT).requires_grad_(True)

    lnz = torch.logsumexp(-β.unsqueeze(-1) * w, dim=1)
    dlnz, = torch.autograd.grad(lnz, β, grad_outputs=torch.ones(β.shape[0]),create_graph=True)
    d2lnz, = torch.autograd.grad(dlnz, β, grad_outputs=torch.ones(β.shape[0]),create_graph=True)

    u = -dlnz
    c = d2lnz * β * β

    return u, c

def measure_mchi0(op : Operator, kT0 : torch.Tensor,
                  B0 : torch.Tensor = torch.tensor([]),
                  field_func:Callable[[torch.Tensor], torch.Tensor]=build_fieldx,
                  B2 : torch.Tensor = torch.tensor([]),
                  B4 : torch.Tensor = torch.tensor([]),
                  B6 : torch.Tensor = torch.tensor([])):
    B1 = field_func(B0)
    w, _ = solve(op, B1, B2, B4, B6)
    β = (1 / kT0)

    lnz = torch.logsumexp(-β.unsqueeze(-1) * w, dim=1)
    dlnz, = torch.autograd.grad(lnz, B0, create_graph=True)
    d2lnz, = torch.autograd.grad(dlnz, B0, create_graph=True)

    m = kT0 * dlnz
    chi = kT0 * d2lnz
    return m, chi

# measure_mchi = torch.vmap(measure_mchi0, in_dims=(None, 0, None, None, None, None, None))
#
# def measure0(op : Operator, kT0 : torch.Tensor,
#                   B0 : torch.Tensor = torch.tensor([]),
#                   field_func:Callable[[torch.Tensor], torch.Tensor]=build_fieldx,
#                   B2 : torch.Tensor = torch.tensor([]),
#                   B4 : torch.Tensor = torch.tensor([]),
#                   B6 : torch.Tensor = torch.tensor([])):
#     B1 = field_func(B0)
#     w, _ = solve(op, B1, B2, B4, B6)
#     β = (1 / kT0)
#     print(β.shape)
#
#     lnz = torch.logsumexp(-β.unsqueeze(-1) * w, dim=1)
#     return lnz
#
# measure = torch.vmap(measure0, in_dims=(None, 0, None, None, None, None, None))
#
# def measure_mchi(op : Operator, kT : torch.Tensor,
#                  B0 : torch.Tensor = torch.tensor([]),
#                  field_func:Callable[[torch.Tensor], torch.Tensor]=build_fieldx,
#                  B2 : torch.Tensor = torch.tensor([]),
#                  B4 : torch.Tensor = torch.tensor([]),
#                  B6 : torch.Tensor = torch.tensor([])):
#
#     B0 = (B0).requires_grad_(True)
#
#     lnz = measure(op, kT.unsqueeze(-1), B0, field_func, B2, B4, B6)
#     print(lnz)
#     dlnz, = torch.autograd.grad(lnz, B0, grad_outputs=torch.ones_like(lnz), create_graph=True)
#     d2lnz, = torch.autograd.grad(dlnz, B0, grad_outputs=torch.ones_like(lnz), create_graph=True)
#
#     m = kT * dlnz
#     chi = kT * d2lnz
#     return m, chi

def measure_mchi(op : Operator, kT : torch.Tensor,
                 B0 : torch.Tensor = torch.tensor([]),
                 field_func:Callable[[torch.Tensor], torch.Tensor]=build_fieldx,
                 B2 : torch.Tensor = torch.tensor([]),
                 B4 : torch.Tensor = torch.tensor([]),
                 B6 : torch.Tensor = torch.tensor([])):
    r"""
    measure the energy and specific heat with steven operators
    Args:
        op (Operator): the steven operator
        kT (torch.Tensor): the temperature
        B0 (torch.Tensor): the magnetic field
        field_func (Callable[[torch.Tensor], torch.Tensor]): the field function to map B0 -> g μB B0
        B2 (torch.Tensor): the coefficient for O[2,m]
        B4 (torch.Tensor): the coefficient for O[4,m]
        B6 (torch.Tensor): the coefficient for O[6,m]
    Returns:
        m (torch.Tensor): the magnetization
        chi (torch.Tensor): the magnetic susceptibility
    """
    num = kT.shape[0]
    m = torch.zeros(num)
    chi = torch.zeros(num)
    B = (B0).requires_grad_(True)
    for id, kT0 in enumerate(kT):
        m[id], chi[id] = measure_mchi0(op, kT0.unsqueeze(-1), B, field_func, B2, B4, B6)
    return m, chi

def fun_lossc(a: torch.Tensor,
              kT: torch.Tensor,
              B0: torch.Tensor,
              field_func:Callable[[torch.Tensor], torch.Tensor],
              cexp: torch.Tensor,
              op : Operator,
              CEFparam_func):
    r"""
    the loss of the reference specific heat

    Args:
        a (torch.Tensor): the parameters tensor
        kT (torch.Tensor): the temperature
        B0 (torch.Tensor): the magnetic field
        field_func (Callable[[torch.Tensor], torch.Tensor]): the field function to map B0 -> g μB B0
        cexp (torch.Tensor): the heat specific
        op (Operator): the steven operators
        CEFparam_func (function): the function map a->B, where B is coefficient for steven operators

    Returns:
        loss (torch.Tensor): the loss.
        dloss (torch.Tensor): the derivative of the loss.

    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    else:
        a = a.clone().detach().requires_grad_(True)
    B2, B4, B6 = CEFparam_func(a)

    _, c = measure_uc(op, kT, B0, field_func, B2, B4, B6)

    loss = torch.sum((c - cexp) ** 2)
    dloss, = torch.autograd.grad(loss, a)
    return loss.detach().numpy(), dloss.detach().numpy()

def fun_losschi(a: torch.Tensor,
                kT: torch.Tensor,
                B0: torch.Tensor,
                field_func:Callable[[torch.Tensor], torch.Tensor],
                chiexp: torch.Tensor,
                op : Operator,
                CEFparam_func):
    r"""
    the loss of the reference specific heat
    the effective magnetic susceptibility is given by
    $$\chi_{{\rm eff}}=\frac{\chi\left(T\right)}{1-\lambda\chi\left(T\right)}+\chi_{0}$$

    Args:
        a (torch.Tensor): the parameters tensor, including CEF parameters and $\lambda$ and $\chi_{0}$
        kT (torch.Tensor): the temperature
        B0 (torch.Tensor): the magnetic field
        field_func (Callable[[torch.Tensor], torch.Tensor]): the field function to map B0 -> g μB B0
        chiexp (torch.Tensor): the magnetic susceptibility
        op (Operator): the steven operators
        CEFparam_func (function): the function map a->B, where B is coefficient for steven operators

    Returns:
        loss (torch.Tensor): the loss.
        dloss (torch.Tensor): the derivative of the loss.
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    else:
        a = a.clone().detach().requires_grad_(True)
    num = a.shape[0]
    B2, B4, B6 = CEFparam_func(a[:num-2])
    lam = a[-2]
    chi0 = a[-1]

    _, chi = measure_mchi(op, kT, B0, field_func, B2, B4, B6)

    chip = chi / (1 - lam * chi) + chi0
    loss = torch.sum((chip - chiexp) ** 2)
    dloss, = torch.autograd.grad(loss, a)

    return loss.detach().numpy(), dloss.detach().numpy()

def fun_loss(a: torch.Tensor,
             field_func,
             op,
             CEFparam_func,
             cdata: Tuple[MeaData, ...] = (),
             chidata : Tuple[MeaData, ...] = (),
             weight = 1.0):
    r"""
    a function to calculate the loss function of specific heat, susceptibility
    Args:
        a (torch.Tensor): the parameters tensor, including CEF parameters and $\lambda$ and $\chi_{0}$
        field_func (Callable[[torch.Tensor], torch.Tensor]): the field function to map B0 -> g μB B0
        op (Operator): the steven operators
        CEFparam_func (function): the function map a->B, where B is coefficient for steven operators
        cdata (Tuple[MeaData, ...]): the cv data to fit
        chidata (Tuple[MeaData, ...]): the chi data to fit
        weight : the weight of the loss function
    Returns:
        loss (torch.Tensor): the loss.
        dloss (torch.Tensor): the derivative of the loss.

    """
    loss = np.array([0.0])
    dloss = np.zeros_like(a)
    num = a.shape[0]
    clen = len(cdata)
    chilen = len(chidata)
    fitlen = clen + chilen

    if weight == 1.0:
        w = np.ones(fitlen)
    else:
        if isinstance(weight, torch.Tensor):
            w = weight.detach().numpy()
        elif isinstance(weight, np.ndarray):
            w = weight
        else:
            raise ValueError("weight must be torch.Tensor or np.ndarray")
        if w.size != fitlen:
            raise ValueError("weight size must be equal to fitlen")

    aCEF = a[:num - 2]
    if len(cdata) != 0:
        for id, chidata0 in enumerate(cdata):
            loss0, dloss0 = fun_lossc(aCEF,
                                      chidata0.kT,
                                      chidata0.B0,
                                      field_func,
                                      chidata0.measure,
                                      op,
                                      CEFparam_func)
            loss += w[id] * loss0
            dloss[:num - 2] += w[id] * dloss0

    if len(chidata) != 0:
        for id, chidata0 in enumerate(chidata):
            loss0, dloss0 = fun_losschi(a,
                                        chidata0.kT,
                                        chidata0.B0,
                                        field_func,
                                        chidata0.measure,
                                        op,
                                        CEFparam_func)
            loss += w[id + clen] * loss0
            dloss += w[id + clen] * dloss0

    return loss, dloss

def fit(a0,
        field_func,
        op,
        CEFparam_func,
        cdata: Tuple[MeaData, ...] = (),
        chidata : Tuple[MeaData, ...] = (),
        weight = 1.0,
        method='L-BFGS-B', bounds=None):
    r"""
        a function to fit specific heat and susceptibility
        Args:
            a0: the initial parameters, including CEF parameters and $\lambda$ and $\chi_{0}$
            field_func (Callable[[torch.Tensor], torch.Tensor]): the field function to map B0 -> g μB B0
            op (Operator): the steven operators
            CEFparam_func (function): the function map a->B, where B is coefficient for steven operators
            cdata (Tuple[MeaData, ...]): the cv data to fit
            chidata (Tuple[MeaData, ...]): the chi data to fit
            weight : the weight of the loss function
            method (str): the optimization method
            bounds (Tuple[float, float]): the bounds of the optimization
        Returns:
            loss (torch.Tensor): the loss.
            dloss (torch.Tensor): the derivative of the loss.

    """

    if bounds is None:
        bounds = list(zip(-np.ones(a0.shape[0]), np.ones(a0.shape[0])))

    res = scipy.optimize.minimize(fun_loss,
                                  a0,
                                  args=(field_func, op, CEFparam_func, cdata, chidata, weight),
                                  method=method,
                                  bounds=bounds,
                                  jac=True, tol=1e-20)
    return res