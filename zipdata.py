import torch
import numpy as np
from typing import Optional

class MeaData:
    kT: Optional[torch.Tensor]
    measure: Optional[torch.Tensor]
    B0: Optional[torch.Tensor]

    def __init__(self):
        self.kT = None
        self.measure = None
        self.B0 = None

    def create(self, kT: torch.Tensor, measure: torch.Tensor, B0: Optional[torch.Tensor|float]):
        r"""
        create the data from list
        Args:
            kT (torch.Tensor): the temperature
            measure (torch.Tensor): measurement
            B0 (Optional[torch.Tensor|float]): the magnetic field

        Returns:

        """
        if isinstance(B0, float):
            B0 = torch.tensor([B0])

        if kT.shape[0] != measure.shape[0]:
            raise ValueError("kT and measure must have same shape")
        self.kT = kT
        self.measure = measure
        self.B0 = B0
        return

    def read(self, filename: str, B0: Optional[torch.Tensor|float]):
        r"""
        create the data from file
        Args:
            filename (str): #(1):kT (2):c
            B0 (Optional[torch.Tensor|float]): magnetic field
        """
        # TODO: check the header of the file
        if isinstance(B0, float):
            B0 = torch.tensor([B0])

        data = np.loadtxt(filename)
        self.kT = torch.from_numpy(data[:, 0])
        self.measure = torch.from_numpy(data[:, 1])
        self.B0 = B0

        return

    def write(self, var:str='cv'):
        if var == 'cv':
            filename = f"cv-{self.B0[0]:.2f}.dat"
        elif var == 'chi':
            filename = f"chi-{self.B0[0]:.2f}.dat"
        else:
            raise ValueError("unknown variable")

        np.savetxt(filename, np.c_[self.kT.detach().numpy(), self.measure.detach().numpy()], delimiter='\t', header='(1)\t(2)\nkT\tc')
        print(f"data is successfully written into {filename}")

        return
