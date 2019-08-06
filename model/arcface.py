import torch
import torch.nn as nn
import math

class Arcface(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=64., m=0.5):
        super(Arcface, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.kernel.data.uniform_(-1,1).renorm_(2, 1, 1e-5).mul_(1e-5)
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m
        self.threshold = math.cos(math.pi - m)

    def forward(self, x, label):
        nB = len(x)
        kernel_norm = torch.div(self.kernel, torch.norm(self.kernel, 2, 0, True))
        cos_theta = torch.mm(x, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1- cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s
        return output
