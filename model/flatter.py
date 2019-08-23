import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatter(nn.Module):
    def __init__(self, embedding_size, init_weights=False):
        super(Flatter, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.conv = nn.Conv2d(512, 512, 7)
        self.prelu = nn.PReLU(512)
        self.linear = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.prelu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)

        # L2 Normalization
        x = torch.div(x, torch.norm(x, 2, 1, True))
        
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
