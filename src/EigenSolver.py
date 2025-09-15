import torch

def lorentzian(x, epsilon = 1e-15):
    return x / (x ** 2 + epsilon)

class EigenSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        w, u = torch.linalg.eigh(A)
        ctx.save_for_backward(w, u)
        return w, u

    @staticmethod
    def backward(ctx, dw, du):
        w, u = ctx.saved_tensors

        F = w - w[:, None]
        F = lorentzian(F)
        F.diagonal().fill_(0)

        udu = u.T.conj() @ du

        return u @ (torch.diag(dw) + 0.5 * F * (udu - udu.T.conj())) @ u.T.conj()

