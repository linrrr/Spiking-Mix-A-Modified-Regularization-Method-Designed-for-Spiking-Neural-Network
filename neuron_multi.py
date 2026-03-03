import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFNeuron(nn.Module):
    def __init__(self, tau, v_th, sigmoid_alpha, elu_alpha, factor_w, factor_b):
        super(LIFNeuron, self).__init__()
        self.v = 0.
        self.tau = tau
        self.v_th = v_th
        self.sigmoid_alpha = sigmoid_alpha
        self.elu_alpha = elu_alpha
        self.factor_w = factor_w
        self.factor_b = factor_b

    def forward(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x).to(x.device)

        output_v = self.v + (x - self.v) / self.tau
        out_v_th = (output_v > self.v_th).float()
        grad_v = F.elu(self.factor_w * torch.sigmoid(self.sigmoid_alpha * (output_v - self.v_th)) - self.factor_b, self.elu_alpha)

        v = (1 - out_v_th) * output_v
        v = v + (x - output_v) / self.tau
        v = (1 - (v > self.v_th).float()) * (v + output_v) / self.tau
        v = output_v + (x - v) / self.tau
        v = (1 - out_v_th) * (v - output_v) / self.tau
        v = v + (x - v) / self.tau

        out_v = (v > self.v_th).float()
        out = (1 - out_v_th) * out_v + out_v_th
        self.v = (1 - out) * output_v

        return (out - grad_v).detach_() + grad_v

    def reset(self):
        self.v = 0.
