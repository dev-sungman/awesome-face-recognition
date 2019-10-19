import torch
import torch.nn as nn
import torch.nn.functional as F

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
        print(cos) 
        # for numerical stability
        cos.clamp(-1, 1)
        
        # for eliminate element w_i = w_j
        cos1 = cos.detach()
        cos1.scatter_(1, torch.arange(self.out_features).view(-1, 1).long(), -100)
        
        _,indices = torch.max(cos1, dim=0)
        mask = torch.zeros((self.out_features, self.out_features))
        print(indices.view(-1,1).long(), indices.view(-1,1).shape)
        mask.scatter_(1, indices.view(-1,1).long(), 1)
        
        exclusive_loss = torch.dot(cos.view(cos.numel()), mask.view(mask.numel())) / self.out_features


if __name__ == '__main__':
    x = torch.randn([512, 10486])
    regularFace = RegularFace(512, 10486)
    y = regularFace(x)
