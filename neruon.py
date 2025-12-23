import torch.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

# v_th, tau, redefine_threshold: These required values need to be calculated.
class LIFNeuron(nn.Module):
    def __init__(self, v_th, tau, redefine_threshold):
        super(LIFNeuron, self).__init__()
        self.v = 0.
        self.v_th = v_th
        self.tau = tau
        self.redefine_threshold = redefine_threshold

    def forward(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x).to(x.device)

        output_v = self.v + (x - self.v) / self.tau
        gard_v = torch.clamp(output_v - self.v_th, 0., 1.)
        out = (output_v > self.v_th).float() * self.redefine_weight(output_v - self.v_th)
        self.v = (1 - ((output_v - self.v_th) > 0.).float()) * output_v

        return (out - gard_v).detach_() + gard_v

    def redefine_weight(self, x):
        weight_value = torch.sigmoid(torch.relu(x))
        return (weight_value > self.v_th).float() * self.redefine_threshold

    def reset(self):
        self.v = 0.

class SharedNeuron(nn.Module):
    def __init__(self):
        super(SharedNeuron, self).__init__()
        self.neuron1 = LIFNeuron(v_th, tau, redefine_threshold)
        self.neuron2 = LIFNeuron(v_th, tau, redefine_threshold)

    def forward(self, x):
        return self.neuron1(x) + self.neuron2(x)

    def reset(self):
        self.neuron1.reset()
        self.neuron2.reset()
