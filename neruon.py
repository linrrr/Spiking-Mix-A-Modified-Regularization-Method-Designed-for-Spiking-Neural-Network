import torch.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFNode(nn.Module):
    def __init__(self):
        super(LIFNode, self).__init__()
        self.v = 0.
        self.v_th = 0.5
        self.tau = 2.0

    def forward(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x).detach().to(x.device)

        output_v = self.v_mul(x, self.v.data)
        oup_v = torch.clamp(output_v - self.v_th, 0., 1.)
        out = self.encoder(output_v, self.v_th) + self.encoder(output_v, self.v_th / self.tau)
        out = self.decoder(out)
        self.v = (1 - ((output_v - self.v_th) > 0.).float()) * output_v

        return (out - oup_v).detach_() + oup_v

    def v_mul(self, x, v_t):
        v = v_t + (x - v_t) / self.tau
        return v

    def encoder(self, v, v_th):
        return (v > v_th).float()

    def decoder(self, x):
        x = (x > self.v_th).float() * self.v_th + (x > self.v_th / self.tau).float() * (self.v_th ** 2)
        return x

    def reset(self):
        self.v = 0.
