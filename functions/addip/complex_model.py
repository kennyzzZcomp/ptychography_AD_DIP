"""Complex backbone + unchanged real amplitude/cos-sin readout for paper net."""
import torch
from torch import nn
from functions.common.complex_layers import (
    ComplexDoubleConv, ComplexMaxPool2d, ComplexConvTranspose2d, complex_cat)


class ComplexProPtyUNet(nn.Module):
    def __init__(self, in_ch, base=23, activation='modrelu'):
        super().__init__()
        if base < 1:
            raise ValueError('complex base channels must be positive')
        c = [base, base*2, base*4, base*8]
        block = lambda a, b: ComplexDoubleConv(a, b, activation)
        self.e1, self.e2, self.e3 = block(in_ch,c[0]), block(c[0],c[1]), block(c[1],c[2])
        self.bot = block(c[2],c[3])
        self.pool = ComplexMaxPool2d()
        self.u3, self.u2, self.u1 = (ComplexConvTranspose2d(c[3],c[2]),
                                    ComplexConvTranspose2d(c[2],c[1]),
                                    ComplexConvTranspose2d(c[1],c[0]))
        self.d3, self.d2, self.d1 = block(c[3],c[2]), block(c[2],c[1]), block(c[1],c[0])
        self.head_amp = nn.Conv2d(2*base, 1, 3, padding=1)
        self.head_phs = nn.Conv2d(2*base, 2, 3, padding=1)

    def forward(self, x):
        x1 = self.e1((x, torch.zeros_like(x)))
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        y = self.d3(complex_cat(self.u3(self.bot(self.pool(x3))), x3))
        y = self.d2(complex_cat(self.u2(y), x2))
        y = self.d1(complex_cat(self.u1(y), x1))
        features = torch.cat(y, 1)
        return self.head_amp(features)[0], self.head_phs(features)[0]
