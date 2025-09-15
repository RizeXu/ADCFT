import torch

# TODO : complete the steven operators
class Operator:
    r"""
    steven operators
    """
    def __init__(self, S: float):
        self.O1 = None
        self.O2 = None
        self.O4 = None
        self.O6 = None
        self.S = S
        self.N = int(2.0 * S) + 1

        assert self.N - 1 - 2.0 * S < 1.0e-5, "invalid S"

        Sp = torch.zeros((self.N, self.N), dtype=torch.complex64)
        Sm = torch.zeros((self.N, self.N), dtype=torch.complex64)
        Sz = torch.diag(torch.arange(self.S, -self.S - 1.0, -1.0)) + 0.0j

        assert(Sz.shape == (self.N, self.N))

        # Sp, Sm
        m = torch.arange(self.S, -self.S - 1.0, -1.0) + 0.0j
        idx = torch.arange(1, self.N, dtype=torch.int32)
        Sp[idx - 1, idx] = torch.sqrt((self.S - m[idx]) * (self.S + m[idx] + 1.0))

        idx = torch.arange(self.N - 1, dtype=torch.int32)
        Sm[idx + 1, idx] = torch.sqrt((self.S + m[idx]) * (self.S - m[idx] + 1.0))

        self.Sp = Sp
        self.Sm = Sm

        self.Sx = 0.5 * (self.Sp + self.Sm)
        self.Sy = -0.5j * (self.Sp - self.Sm)
        self.Sz = Sz

        self.build_StevenOp()
        print(f"the steven operator for s={self.S} is successfully built")

    def build_StevenOp(self):
        """
        col = l - m

        only l <= 2s is valid
        """
        X = self.S * (self.S + 1.0)
        I = torch.diag(torch.ones(self.N, dtype=torch.complex64))
        Sx = self.Sx
        Sy = self.Sy
        Sz = self.Sz
        Sp = self.Sp
        Sm = self.Sm

        Sp1 = Sp
        Sp2 = Sp @ Sp
        Sp3 = Sp2 @ Sp
        Sp4 = Sp3 @ Sp
        Sp5 = Sp4 @ Sp
        Sp6 = Sp5 @ Sp
        Sm1 = Sm
        Sm2 = Sm @ Sm
        Sm3 = Sm2 @ Sm
        Sm4 = Sm3 @ Sm
        Sm5 = Sm4 @ Sm
        Sm6 = Sm5 @ Sm
        Sz2 = Sz @ Sz
        Sz3 = Sz2 @ Sz
        Sz4 = Sz3 @ Sz
        Sz5 = Sz4 @ Sz
        Sz6 = Sz5 @ Sz

        # O1
        O1 = [Sx, Sz, Sy]
        self.O1 = torch.stack(O1, dim=0)

        # O2
        O2 = [Sx @ Sx - Sy @ Sy,
              0.5 * (Sz @ Sx + Sx @ Sz),
              3.0 * Sz2 - X * I,
              0.5 * (Sy @ Sz + Sz @ Sy),
              Sx @ Sy + Sy @ Sx]
        self.O2 = torch.stack(O2, dim=0)

        # O4
        C42 = 7.0 * Sz2 - (X + 5.0) * I
        C41 = 7.0 * Sz3 - (3.0 * X + 1.0) * Sz

        O4 = [0.5 * (Sp4 + Sm4),
              0.25 * ((Sp3 + Sm3) @ Sz + Sz @ (Sp3 + Sm3)),
              0.25 * ((Sp2 + Sm2) @ C42 + C42 @ (Sp2 + Sm2)),
              0.25 * ((Sp1 + Sm1) @ C41 + C41 @ (Sp1 + Sm1)),
              35.0 * Sz4 - (30.0 * X - 25.0) * Sz2 + (3.0 * X * X - 6.0 * X) * I,
              -0.25j * ((Sp1 - Sm1) @ C41 + C41 @ (Sp1 - Sm1)),
              -0.25j * ((Sp2 - Sm2) @ C42 + C42 @ (Sp2 - Sm2)),
              -0.25j * ((Sp3 - Sm3) @ Sz + Sz @ (Sp3 - Sm3)),
              -0.5j * (Sp4 - Sm4)]

        self.O4 = torch.stack(O4, dim=0)

        # O6
        C64 = 11.0 * Sz2 - (X + 38.0) * I
        C63 = 11.0 * Sz3 - (3.0 * X + 59.0) * Sz
        C62 = 33.0 * Sz4 - (18.0 * X + 123.0) * Sz2 + (X * X + 10.0 * X + 102.0) * I
        C61 = 33.0 * Sz5 - (30.0 * X - 15.0) * Sz3 + (5.0 * X * X - 10.0 * X + 12.0) * Sz

        O6 = [0.5 * (Sp6 + Sm6),
              0.25 * ((Sp5 + Sm5) @ Sz + Sz @ (Sp5 + Sm5)),
              0.25 * ((Sp4 + Sm4) @ C64 + C64 @ (Sp4 + Sm4)),
              0.25 * ((Sp3 + Sm3) @ C63 + C63 @ (Sp3 + Sm3)),
              0.25 * ((Sp2 + Sm2) @ C62 + C62 @ (Sp2 + Sm2)),
              0.25 * ((Sp1 + Sm1) @ C61 + C61 @ (Sp1 + Sm1)),
              231.0 * Sz6 - (315.0 * X - 735.0) * Sz4
              + (105.0 * X * X - 525.0 * X + 294.0) * Sz2
              - (5.0 * X * X * X - 40.0 * X * X + 60.0 * X) * I,
              -0.25j * ((Sp1 - Sm1) @ C61 + C61 @ (Sp1 - Sm1)),
              -0.25j * ((Sp2 - Sm2) @ C62 + C62 @ (Sp2 - Sm2)),
              -0.25j * ((Sp3 - Sm3) @ C63 + C63 @ (Sp3 - Sm3)),
              -0.25j * ((Sp4 - Sm4) @ C64 + C64 @ (Sp4 - Sm4)),
              -0.25j * ((Sp5 - Sm5) @ Sz + Sz @ (Sp5 - Sm5)),
              -0.5j * (Sp6 - Sm6)]

        self.O6 = torch.stack(O6, dim=0)

    def show(self):
        print("========================================")
        for m, O in enumerate(self.O2):
            print(f"O[2,{2 - m}]")
            print(O.data)
        print("========================================")
        for m, O in enumerate(self.O4):
            print(f"O[4,{4 - m}]")
            print(O.data)
        print("========================================")
        for m, O in enumerate(self.O6):
            print(f"O[6,{6 - m}]")
            print(O.data)
        print("========================================")
