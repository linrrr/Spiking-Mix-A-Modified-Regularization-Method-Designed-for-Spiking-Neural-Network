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
            self.v = torch.zeros_like(x).to(x.device)

        v = self.v + (x - self.v) / self.tau
        oup = torch.clamp(v - self.v_th, 0., 1.)
        out = self.v_th * (v > self.v_th).float() + 0.5 * self.v_th * (v > 0.).float()  
        self.v = ((v - self.v_th) < 0.).float() * v
        return (out - oup).detach_() + oup

    def reset(self):
        self.v = 0.
