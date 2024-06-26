import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

class DnCNNwithRL_SR(nn.Module):
    def __init__(self, depth=17, in_channels=64, image_channels=3, kernel_size=3, upscale_factor=2):
        super(DnCNNwithRL_SR, self).__init__()
        padding = 1
        self.upscale_factor = upscale_factor
        layers = []
        
        # First convolution layer
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Add residual blocks
        for _ in range(depth - 2):
            layers.append(ResidualBlock(in_channels, kernel_size=kernel_size, padding=padding))
        
        # Last convolution layer
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=image_channels * (upscale_factor ** 2), kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.PixelShuffle(upscale_factor))  # Upscale the image
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dncnn(x)
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False) # Resize the image to the original size

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def create_DnCNNwithRL_SR(depth=17, in_channels=64, image_channels=3, kernel_size=3, upscale_factor=2):
    return DnCNNwithRL_SR(depth=depth, in_channels=in_channels, image_channels=image_channels, kernel_size=kernel_size, upscale_factor=upscale_factor)
