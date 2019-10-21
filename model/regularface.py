import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class RegularFace(nn.Module):
    def __init__(self, in_features, out_features):
        super(RegularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos = torch.mm(weight_norm, weight_norm.t())
        
        # for numerical stability
        cos.clamp(-1, 1)
        
        # for eliminate element w_i = w_j
        cos_ind = cos.detach()

        ind = np.diag_indices(cos.shape[0])
        min_ind = torch.min(cos_ind) - 1
        cos_ind[ind[0], ind[1]] = torch.full((cos_ind.shape[0],), min_ind)

        _,indices = torch.max(cos_ind, dim=0)
        mask = torch.zeros((self.out_features, self.out_features))
        mask.scatter_(1, indices.view(-1,1).long(), 1)
        
        exclusive_loss = torch.dot(cos.view(cos.numel()), mask.view(mask.numel())) / self.out_features
        
        return exclusive_loss

if __name__ == '__main__':
    x = torch.randn([512, 10486])
    regularFace = RegularFace(512, 10486)
    y = regularFace(x)
