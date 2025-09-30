import torch
import numpy as np
import scipy
from typing import Optional, Tuple

from src.Operator import Operator
from src.unit import unit
from src.zipdata import MeaData
from src.solve import solve, measure_uc, measure_mchi
from src.utils import print_state

zero = torch.tensor([0.0])

# TODO : measure SF
class Model:
    cdata: Tuple[MeaData,...]
    chidata: Tuple[MeaData,...]
    insdata: Tuple[MeaData,...]

    def __init__(self, spin: float, g : torch.Tensor):
        r"""
        build a model
        Args:
            spin (float): spin
            g (torch.Tensor): g-tensor
        """
        self.spin = spin
        self.g = g
        self.op = Operator(spin)
        self.λ = torch.tensor([0.0])
        self.chi0 = torch.tensor([0.0])

        self.σ = torch.tensor([0.01])
        self.fac = torch.tensor([1.0])

        self.flag_aCEF = False
        self.aCEF = torch.tensor([])
        self.B2 = torch.tensor([])
        self.B4 = torch.tensor([])
        self.B6 = torch.tensor([])

        self.cdata = ()
        self.chidata = ()
        self.insdata = ()

    def _build_fieldx(self, B):
        # x-axis
        H = torch.cat([B, zero, zero])
        H = unit.μB * self.g * H
        return H

    def _build_fieldy(self, B):
        # y-axis
        H = torch.cat([zero, zero, B])
        H = unit.μB * self.g * H
        return H

    def _build_fieldz(self, B):
        # z-axis
        H = torch.cat([zero, B, zero])

        H = unit.μB * self.g * H
        return H

    def build_field(self, B, axis:str = 'z'):
        r"""
        The function to build H for z-axis
        Note that for steven operators the sequence is x, z, y !!!
        Args:
            B: the magnetic field
            axis: the axis to build, default is 'z'

        Returns:
            H: g μB B

        """

        if axis == 'x':
            return self._build_fieldx(B)
        elif axis == 'y':
            return self._build_fieldy(B)
        elif axis == 'z':
            return self._build_fieldz(B)
        else:
            raise ValueError(f'axis {axis} is not supported.')

    def set_aCEF(self, B2: torch.Tensor, B4: torch.Tensor, B6: torch.Tensor):
        """
        set the aCEF parameters
        Args:
            B2 (torch.Tensor): the parameters for O2
            B4 (torch.Tensor): the parameters for O4
            B6 (torch.Tensor): the parameters for O6

        """
        if B2.shape[0] != 5:
            raise ValueError('the size of B2 must be 5')
        if B4.shape[0] != 9:
            raise ValueError('the size of B4 must be 9')
        if B6.shape[0] != 13:
            raise ValueError('the size of B6 must be 13')

        self.B2 = B2
        self.B4 = B4
        self.B6 = B6
        self.flag_aCEF = True

        print('set CEF parameters successful')

        return

    def solve(self,
              B0 : torch.Tensor = torch.tensor([0.0]),
              axis : str = 'z',
              in_unit : str = 'exp',
              if_print: bool = True):
        """
        Args:
            B0 (torch.Tensor): the magnetic field
            axis: the axis of magnetic field, default is 'z'
            in_unit: the unit of output in theory (theo) or experiment (exp), default is 'exp'
            if_print (bool) : if print the state

        Returns:
            enr (torch.Tensor): the energy, [K] for 'theo' and [meV] for 'exp'
            psi (torch.Tensor): the coefficient of the state

        """

        if in_unit not in ('theo', 'exp'):
            raise ValueError('the unit of output in theory must be exp or theo')

        if not self.flag_aCEF:
            raise ValueError('the aCEF have not be set, use set_aCEF() or set_a() before solve.')

        enr, psi = solve(self.op, self.build_field(B0, axis), self.B2, self.B4, self.B6)

        if in_unit == 'exp':
            enr = enr / unit.kB

        if if_print:
            print_state(self.op.N, enr, psi)

        return enr, psi

    def measure_uc(self,
                   kT: torch.Tensor,
                   B0 : torch.Tensor = torch.tensor([0.0]),
                   axis : str = 'z',
                   in_unit: str = 'exp'):
        r"""
        measure the internal energy u and specific heat c

        Args:
            kT (torch.Tensor): the temperature
            B0 (torch.Tensor): the magnetic field
            axis (str): the axis of magnetic field, default is 'z'
            in_unit (str): the unit of output in theory (theo) or experiment (exp), default is 'exp'

        Returns:
            u (torch.Tensor): the internal energy, unit : [K]
            c (torch.Tensor): the specific heat, unit : [J K^{-1} mol^{-1}]

        """
        if not self.flag_aCEF:
            raise ValueError('the aCEF have not be set')

        if in_unit not in ('theo', 'exp'):
            raise ValueError('the unit of output in theory must be exp or theo')

        if axis == 'x':
            build_field = self._build_fieldx
        elif axis == 'y':
            build_field = self._build_fieldy
        elif axis == 'z':
            build_field = self._build_fieldz
        else:
            raise ValueError('the axis must be x or y or z')

        u, c = measure_uc(self.op, kT, B0, build_field, self.B2, self.B4, self.B6)

        if in_unit == 'exp':
            u = u / unit.kB
            c = c * unit.R
        return u, c

    def measure_mchi(self,
                     kT: torch.Tensor,
                     B0 : torch.Tensor = torch.tensor([0.0]),
                     axis: str = 'z',
                     eff: bool = True,
                     in_unit: str = 'exp'):
        r"""
        measure the magnetization m and susceptibility chi
        Args:
            kT (torch.Tensor): the temperature
            B0 (torch.Tensor): the magnetic field
            axis (str): the axis of magnetic field, default is 'z'
            eff (bool) : if use the effective susceptibility
            in_unit (str): the unit of output in theory (theo) or experiment (exp), default is 'exp'

        Returns:
            m (torch.Tensor): the magnetization m, unit : [K T^{-1}]
            chi (torch.Tensor): the susceptibility chi, unit : [emu mol^{-1}]

        """
        if not self.flag_aCEF:
            raise ValueError('the aCEF have not be set')

        if in_unit not in ('theo', 'exp'):
            raise ValueError('the unit of output in theory must be exp or theo')

        if axis == 'x':
            build_field = self._build_fieldx
        elif axis == 'y':
            build_field = self._build_fieldy
        elif axis == 'z':
            build_field = self._build_fieldz
        else:
            raise ValueError('the axis must be x or y or z')

        m, chi = measure_mchi(self.op, kT, B0, build_field, self.B2, self.B4, self.B6)
        if in_unit == 'exp':
            m = m / unit.μB / self.op.S
            chi = 0.1 * unit.R * chi

        if eff:
            chi = chi / (1 - self.λ * chi) + self.chi0

        return m, chi

    def build_CEFparam(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def set_a(self, a: torch.Tensor):
        r"""
        set the CEF parameters using the built-in build_CEFparam function
        Args:
            a: the input parameters

        """
        self.B2, self.B4, self.B6 = self.build_CEFparam(a)

        self.flag_aCEF = True
        return

    def set_fitres(self, res):
        r"""
        set the fit results to the model, including CEF parameters, λ and chi0
        Args:
            res: the optimal result

        """
        x = torch.tensor(res.x)
        num = x.shape[0]
        a = x[:num - 2]
        self.set_a(a)

        self.λ = x[-2]
        self.chi0 = x[-1]

        return

    def set_fiteffectiveres(self, res):
        r"""
        set the effective parameters of λ and chi0
        Args:
            res: the optimal result

        """
        x = torch.tensor(res.x)
        assert x.shape[0] == 2, "the number of fit effective parameters should be 2"
        self.λ = x[0]
        self.chi0 = x[1]
        return

    def read_cdata(self,
                   filename: str,
                   B0: Optional[torch.Tensor|float],
                   axis: str = 'z'):
        r"""
        Read the heat specific data from file
        Args:
            filename (str): the filename
            B0 (Optional[torch.Tensor|float]): the magnetic field
            axis (str): the axis of the magnetic field
        """
        cdata_ = MeaData()
        cdata_.read(filename, B0, axis)
        self.cdata = self.cdata + (cdata_,)
        return

    def read_chidata(self,
                     filename: str,
                     B0: Optional[torch.Tensor|float],
                     axis: str = 'z'):
        r"""
        Read the heat specific data from file
        Args:
            filename (str): the filename
            B0 (Optional[torch.Tensor|float]): the magnetic field
            axis (str): the axis of the magnetic field
        """
        chidata_ = MeaData()
        chidata_.read(filename, B0, axis)
        self.chidata = self.chidata + (chidata_,)
        return

    def read_insdata(self,
                     filename: str,
                     T0: Optional[torch.Tensor|float],
                     B0: Optional[torch.Tensor|float],
                     axis: str = 'z'):
        r"""
        Read the intensity data from file
        Args:
            filename (str): the filename
            T0 (Optional[torch.Tensor|float]): the temperature
            B0 (Optional[torch.Tensor|float]): the magnetic field
            axis (str): the axis of the magnetic field
        """
        insdata_ = MeaData()
        insdata_.readINS(filename, T0, B0, axis)
        self.insdata = self.insdata + (insdata_,)
        return

    # def fun_loss

    def fun_lossc(self,
                  a: torch.Tensor,
                  kT: torch.Tensor,
                  B0: torch.Tensor,
                  cexp: torch.Tensor,
                  axis: str = 'z',
                  in_unit: str = 'exp'):
        r"""
        the loss of the reference specific heat

        Args:
            a (torch.Tensor): the parameters tensor
            kT (torch.Tensor): the temperature
            B0 (torch.Tensor): the magnetic field
            cexp (torch.Tensor): the heat specific
            axis (str): the axis of the magnetic field
            in_unit (str): use the unit in theo (theory) or experiment (exp), default is 'exp'

        Returns:
            loss (torch.Tensor): the loss.
            dloss (torch.Tensor): the derivative of the loss.

        """
        if in_unit not in ('theo', 'exp'):
            raise ValueError('the unit of output in the heat specification must be exp or theo')

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        else:
            a = a.clone().detach().requires_grad_(True)

        B2, B4, B6 = self.build_CEFparam(a)

        op = self.op

        if axis == 'x':
            build_field = self._build_fieldx
        elif axis == 'y':
            build_field = self._build_fieldy
        elif axis == 'z':
            build_field = self._build_fieldz
        else:
            raise ValueError('the axis must be x or y or z')

        _, c = measure_uc(op, kT, B0, build_field, B2, B4, B6)

        if in_unit == 'exp':
            c = c * unit.R

        loss = torch.mean((c - cexp) ** 2)

        dloss, = torch.autograd.grad(loss, a)
        return loss.detach().numpy(), dloss.detach().numpy()

    def fun_losschi(self,
                    a: torch.Tensor,
                    kT: torch.Tensor,
                    B0: torch.Tensor,
                    chiexp: torch.Tensor,
                    axis: str = 'z',
                    eff: bool = True,
                    in_unit: str = 'exp'):
        r"""
        the loss of the reference susceptibility
        the effective magnetic susceptibility is given by
        $$\chi_{{\rm eff}}=\frac{\chi\left(T\right)}{1-\lambda\chi\left(T\right)}+\chi_{0}$$

        Args:
            a (torch.Tensor): the parameters tensor, including CEF parameters and $\lambda$ and $\chi_{0}$
            kT (torch.Tensor): the temperature
            B0 (torch.Tensor): the magnetic field
            chiexp (torch.Tensor): the magnetic susceptibility
            axis (str): the axis of the magnetic field
            eff (bool): whether to use the effective susceptibility
            in_unit (str): use the unit in theo (theory) or experiment (exp), default is 'exp'

        Returns:
            loss (torch.Tensor): the loss.
            dloss (torch.Tensor): the derivative of the loss.
        """

        if in_unit not in ('theo', 'exp'):
            raise ValueError('the unit of output in the heat specification must be exp or theo')

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        else:
            a = a.clone().detach().requires_grad_(True)
        num = a.shape[0]
        B2, B4, B6 = self.build_CEFparam(a[:num - 2])
        lam = a[-2]
        chi0 = a[-1]

        op = self.op
        if axis == 'x':
            build_field = self._build_fieldx
        elif axis == 'y':
            build_field = self._build_fieldy
        elif axis == 'z':
            build_field = self._build_fieldz
        else:
            raise ValueError('the axis must be x or y or z')
        _, chi = measure_mchi(op, kT, B0, build_field, B2, B4, B6)

        if in_unit == 'exp':
            chi = 0.1 * unit.R * chi

        if eff:
            chip = chi / (1 - lam * chi) + chi0
        else:
            chip = chi

        loss = torch.mean((chip - chiexp) ** 2)
        dloss, = torch.autograd.grad(loss, a)

        return loss.detach().numpy(), dloss.detach().numpy()

    def fun_losseffective(self,
                          a: torch.Tensor,
                          chi: Tuple[torch.Tensor, ...]):

        r"""
        the loss of the reference susceptibility
        the effective magnetic susceptibility is given by
        $$\chi_{{\rm eff}}=\frac{\chi\left(T\right)}{1-\lambda\chi\left(T\right)}+\chi_{0}$$

        Args:
            a (torch.Tensor): the parameters tensor, including CEF parameters and $\lambda$ and $\chi_{0}$
            chi (torch.Tensor): the original theoretical magnetic susceptibility

        Returns:
            loss (torch.Tensor): the loss.
            dloss (torch.Tensor): the derivative of the loss.
        """

        num = a.shape[0]
        chiexplen = len(self.chidata)

        if chiexplen == 0:
            raise ValueError('no magnetic susceptibility obtained')

        if num != 2:
            raise ValueError('only 2 input is needed')

        if not isinstance(a, torch.Tensor):
            a_ = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        else:
            a_ = a.clone().detach().requires_grad_(True)

        chiexp = torch.stack([data.measure for data in self.chidata])
        chi0 = torch.stack(list(chi))

        chi0_ = chi0 / (1 - a_[0] * chi0) + a_[1]

        loss = torch.mean((chi0_ - chiexp) ** 2)
        dloss, = torch.autograd.grad(loss, a_)

        return loss.detach().numpy(), dloss.detach().numpy()

    def fun_loss(self,
                 a: torch.Tensor,
                 weight=1.0,
                 eff: bool = True,
                 in_unit: str = 'exp'):
        r"""
        a function to calculate the loss function of specific heat, susceptibility
        Args:
            a (torch.Tensor): the parameters tensor, including CEF parameters and $\lambda$ and $\chi_{0}$
            weight : the weight of the loss function
            eff (bool): whether to use the effective susceptibility
            in_unit (str): use the unit in theo (theory) or experiment (exp), default is 'exp'
        Returns:
            loss (torch.Tensor): the loss.
            dloss (torch.Tensor): the derivative of the loss.

        """
        loss = np.array([0.0])
        dloss = np.zeros_like(a)
        num = a.shape[0]
        clen = len(self.cdata)
        chilen = len(self.chidata)
        fitlen = clen + chilen

        if isinstance(weight, torch.Tensor):
            w = weight.detach().numpy()
        elif isinstance(weight, np.ndarray):
            w = weight
        else:
            raise ValueError("weight must be torch.Tensor or np.ndarray")
        if w.size != fitlen:
            raise ValueError("weight size must be equal to fitlen")

        aCEF = a[:num - 2]
        if clen != 0:
            for cdata_id, cdata0 in enumerate(self.cdata):
                loss0, dloss0 = self.fun_lossc(aCEF,
                                               cdata0.kT,
                                               cdata0.B0,
                                               cdata0.measure,
                                               cdata0.axis,
                                               in_unit)
                loss += w[cdata_id] * loss0
                dloss[:num - 2] += w[cdata_id] * dloss0

        if chilen != 0:
            for chidata_id, chidata0 in enumerate(self.chidata):
                loss0, dloss0 = self.fun_losschi(a,
                                                 chidata0.kT,
                                                 chidata0.B0,
                                                 chidata0.measure,
                                                 chidata0.axis,
                                                 eff,
                                                 in_unit)
                loss += w[chidata_id + clen] * loss0
                dloss += w[chidata_id + clen] * dloss0

        return loss, dloss

    def fit(self,
            a0,
            weight = 1.0,
            eff: bool= True,
            in_unit: str = 'exp',
            method='L-BFGS-B',
            bounds=None):
        r"""
        fit the model
        Args:
            a0 (torch.Tensor): the parameters tensor, including CEF parameters and $\lambda$
            weight (torch.tensor): the weight of the loss function of specific heat, susceptibility
            eff (bool): whether to use effective susceptibility, default is True
            in_unit (str): use the unit in theo (theory) or experiment (exp), default is 'exp'
            method (str): the method of fitting, default is 'L-BFGS-B'
            bounds (Tuple[float, float]): the bounds of the parameters, default is None

        Returns:
            res : the fit results

        """

        loss0, _ = self.fun_loss(a0, weight)
        print("with initial loss :", loss0)

        # TODO: is this vaild
        # if bounds is None:
        #     bounds = list(zip(-np.ones(a0.shape[0]), np.ones(a0.shape[0])))

        res = scipy.optimize.minimize(self.fun_loss,
                                      a0,
                                      args=(weight, eff, in_unit),
                                      method=method,
                                      bounds=bounds,
                                      jac=True, tol=1e-20)
        return res

    def fit_effectivechi(self,
                         a0,
                         in_unit: str = 'exp',
                         method='L-BFGS-B',
                         bounds=None):
        r"""
        fit the effective parameters of the model
        Args:
            a0 (torch.Tensor): the parameters tensor, $\lambda$ and $\chi_{0}$
            in_unit (str): use the unit in theo (theory) or experiment (exp), default is 'exp'
            method (str): the method of fitting, default is 'L-BFGS-B'
            bounds (Tuple[float, float]): the bounds of the parameters, default is None

        Returns:
            res : the fit results

        """

        if not self.flag_aCEF:
            raise ValueError("aCEF is not set")

        if in_unit not in ('theo', 'exp'):
            raise ValueError('the unit of output in the heat specification must be exp or theo')

        chi = ()

        for chidata_id, chidata0 in enumerate(self.chidata):
            _, chi0 = self.measure_mchi(chidata0.kT,
                                        chidata0.B0,
                                        chidata0.axis,
                                        eff=False,
                                        in_unit=in_unit)

            chi = chi + (chi0,)

        loss, _ = self.fun_losseffective(a0, chi)
        print("initial loss :", loss)

        res = scipy.optimize.minimize(self.fun_losseffective,
                                      a0,
                                      args=chi,
                                      method=method,
                                      bounds=bounds,
                                      jac=True, tol=1e-20)
        return res

if __name__ == '__main__':
    model = Model(4.5, torch.tensor([1.0, 1.0, 1.0]))
