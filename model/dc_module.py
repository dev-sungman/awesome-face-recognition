import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, device, stride=1, padding=1, magnitude=None, angular=None):
        super(Conv2d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.padding = padding
        self.k_size = k_size
        self.kernel = self._get_conv_filter(out_ch, in_ch, k_size)
        self.eps = 1e-4
        self.device = device

        #self.radius = nn.Parameter(torch.Tensor(1, 1, 1, 1)).to(self.device)
        #self.radius = nn.init.constant_(self.radius, 1.0) ** 2 + self.eps
        self.radius = self._get_radius(in_ch=1)
        
        self.magnitude = magnitude
        self.angular = angular
    
    def _get_conv_filter(self, out_ch, in_ch, k_size):
        kernel = nn.Parameter(torch.Tensor(out_ch, in_ch, k_size, k_size))
        
        # weight intialization
        kernel = nn.init.kaiming_normal_(kernel)

        return kernel
    
    def _get_radius(self, in_ch):
        radius = nn.Parameter(torch.Tensor(in_ch, 1, 1, 1)).to(self.device)
        radius = nn.init.constant_(radius, 1.0) ** 2 + self.eps

        return radius

    def _get_filter_norm(self, kernel):
        return torch.norm(kernel.data, 2, 1, True)
    
    def _get_input_norm(self, feat):
        #print(feat.shape) 
        f = torch.ones(1, self.in_ch, self.k_size, self.k_size).to(self.device)
        #print('f: ', f.shape)
        input_norm = torch.sqrt(F.conv2d(feat*feat, f, stride=self.stride, padding=self.padding)+self.eps)
        return input_norm

    def get_orth_constraint(self, filt, nfilt):
        filt = filt.view(-1, nfilt)

        dot_product = torch.matmul(filt.t(), filt)
        
        #criterion = nn.MSELoss()
        loss = 1e-5 * torch.norm(dot_product - torch.eye(nfilt).to(self.device), 2)
        return loss
        
    def forward(self, x):
        if self.magnitude is None:
            out = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)

        elif self.magnitude == "ball":
            x_norm = self._get_input_norm(x) 
            
            # weight normalization
            w_norm = self._get_filter_norm(self.kernel)
            kernel_tensor = nn.utils.parameters_to_vector(self.kernel)
            kernel_tensor = torch.reshape(kernel_tensor,self.kernel.shape)
            kernel_tensor = kernel_tensor / w_norm
            
            out = F.conv2d(x, kernel_tensor, stride=self.stride, padding=self.padding)
            
            x_norm_mean = torch.mean(x_norm)
            adj_radius = self.radius * x_norm_mean
            min_x_radius = torch.min(x_norm, adj_radius)

            out = (out / x_norm) * (min_x_radius / adj_radius)

        elif self.magnitude == "linear":
            
            x_norm = self._get_input_norm(x) 
            w_norm = self._get_filter_norm(self.kernel)
            
            kernel_tensor = nn.utils.parameters_to_vector(self.kernel)
            kernel_tensor = torch.reshape(kernel_tensor,self.kernel.shape)
            kernel_tensor = kernel_tensor / w_norm
            
            out = F.conv2d(x, kernel_tensor, stride=self.stride, padding=self.padding)
        
        else:
            out = None

        return out


class DCNet(nn.Module):
    def __init__(self, magnitude, angular, device):
        super(DCNet, self).__init__()
        
        self.features = nn.Sequential(
                Conv2d(in_ch=3, out_ch=6, k_size=5, magnitude=magnitude, device=device),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=2),
                Conv2d(in_ch=6, out_ch=16, k_size=5, magnitude=magnitude, device=device),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=2),
                Conv2d(in_ch=16, out_ch=120, k_size=5, magnitude=magnitude, device=device),
                nn.ReLU()
                )

        self.fc = nn.Sequential(
                nn.Linear(1920, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                )

    def get_orth_loss(self):
        loss = 0
        for layer in self.features.named_children():
            _layer = layer[1]
            layer_type = str(layer[1])
            # if layer is Conv2d, get orthogonal loss
            if layer_type == "Conv2d()":
                loss += _layer.get_orth_constraint(_layer.kernel, _layer.in_ch)

        return loss
    
    def forward(self, x):
        x = self.features(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    
        
            
if __name__ == '__main__':
    net = DCNet(magnitude=None, angular='cos')
    
    x = torch.randn([3,1,27,27])
    y = net(x)



