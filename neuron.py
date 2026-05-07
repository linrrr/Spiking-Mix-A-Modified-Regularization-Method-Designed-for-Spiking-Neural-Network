import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFNeuron(nn.Module):
    def __init__(self, tau, v_th, sigmoid_alpha, elu_alpha, sigmoid_weight, b, w_th):
        super(LIFNeuron, self).__init__()
        self.v = 0.
        self.tau = tau
        self.v_th = v_th
        self.sigmoid_alpha = sigmoid_alpha
        self.elu_alpha = elu_alpha
        self.sigmoid_weight = sigmoid_weight
        self.b = b
        self.w_th = w_th

    def forward(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x).to(x.device)

        output_v = self.v + (x - self.v) / self.tau
        out = (output_v > self.v_th).float()
        grad_v = F.elu(self.sigmoid_weight * torch.sigmoid(self.sigmoid_alpha * (output_v - self.v_th)) + self.b, self.elu_alpha)
        v = self.weight(out, output_v, x)
        out = (v > self.v_th).float()
        self.v = (1 - out) * output_v

        return (out - grad_v).detach_() + grad_v

    def weight(self, out, output_v, x):
        v = (1 - out) * output_v
        v = v + (x - output_v) / self.tau
        v = (1 - (v > self.v_th).float()) * (v + output_v) / self.tau
        v = output_v + (x - v) / self.tau
        v = (1 - out) * (v - output_v) / self.tau
        v = v + (x - v) / self.tau

        weight = self.sigmoid_weight * torch.sigmoid(torch.relu(x)) + self.b
        weight = (weight > self.w_th).float() * self.w_th
        return v + weight * output_v

    def reset(self):
        self.v = 0.
