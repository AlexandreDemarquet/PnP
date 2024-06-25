import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class MultiDownHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiDownHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm([out_channels, out_channels])
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(self.norm(x))
        x = self.conv2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)  # [N, C, H, W] -> [HW, N, C]
        x = self.transformer(x)
        return x.permute(1, 2, 0).view(x.size(1), -1, int(x.size(0)**0.5), int(x.size(0)**0.5))  # [HW, N, C] -> [N, C, H, W]

class GDFN(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(GDFN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)
        gate = torch.sigmoid(self.gate(identity))
        return x * gate
    
class TransposedAttention(nn.Module):
    def __init__(self, in_channels):
        super(TransposedAttention, self).__init__()
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        b, c, h, w = x.size()
        qkv = self.qkv(x).view(b, 3, c, -1).permute(1, 0, 2, 3)  # [B, 3, C, HW]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.softmax(q @ k.transpose(-2, -1) / c**0.5)  # [B, C, HW, HW]
        return (attn @ v).view(b, c, h, w) + x
    

class CGNet(nn.Module):
    def __init__(self):
        super(CGNet, self).__init__()
        self.multi_down_head = MultiDownHead(in_channels=3, out_channels=64)
        
        self.transformer_block_l1 = TransformerBlock(num_layers=2, d_model=64, nhead=8, dim_feedforward=256)
        self.transformer_block_l2 = TransformerBlock(num_layers=2, d_model=128, nhead=8, dim_feedforward=512)
        self.transformer_block_l3 = TransformerBlock(num_layers=2, d_model=256, nhead=8, dim_feedforward=1024)
        self.transformer_block_l4 = TransformerBlock(num_layers=2, d_model=512, nhead=8, dim_feedforward=2048)
        
        self.gdfn = GDFN(in_channels=512, hidden_dim=1024)
        self.transposed_attention = TransposedAttention(in_channels=512)
        
        self.up_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.multi_down_head(x)
        
        x = self.transformer_block_l1(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.transformer_block_l2(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.transformer_block_l3(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.transformer_block_l4(x)
        
        x = self.gdfn(x)
        x = self.transposed_attention(x)
        
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        x = self.final_conv(x)
        return x


def create_CGNet():
    return CGNet()