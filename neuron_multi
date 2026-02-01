import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFNeuron(nn.Module):
    def __init__(self, tau0, tau1, v_th, alpha):
        super(LIFNeuron, self).__init__()
        self.v = 0.
        self.tau0 = tau0
        self.tau1 = tau1
        self.v_th = v_th
        self.alpha = alpha

    def forward(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x).to(x.device)

        output_v = self.v + (x - self.v) / self.tau0
        out = (output_v * self.tau0 > self.v_th).float() / self.tau1 + (output_v > self.v_th).float() / self.tau0
        grad_v = torch.sigmoid(self.alpha * (output_v - self.v_th))
        self.v = (1 - out) * output_v

        return (out - grad_v).detach_() + grad_v

    def reset(self):
        self.v = 0.
