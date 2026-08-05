import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==============================================================================
# Implementation note.
# ==============================================================================
class TemporalConv(nn.Module):
    def __init__(self, in_chans=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 5, kernel_size=(1,15), stride=(1,5), padding=(0,7))
        self.norm1 = nn.GroupNorm(1, 5)
        self.conv2 = nn.Conv2d(5, 8, kernel_size=(1,15), stride=(1,1), padding=(0,7))
        self.norm2 = nn.GroupNorm(1, 8)
        self.conv3 = nn.Conv2d(8, 10, kernel_size=(1,15), stride=(1,1), padding=(0,7))
        self.norm3 = nn.GroupNorm(1, 10)
        self.conv4 = nn.Conv2d(10, 16, kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.norm4 = nn.GroupNorm(1, 16)
        self.conv5 = nn.Conv2d(16, 20, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.norm5 = nn.GroupNorm(1, 20)
        self.conv6 = nn.Conv2d(20, 25, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.norm6 = nn.GroupNorm(1, 25)
        self.gelu = nn.GELU()

    def forward(self, x):  # [B, 16, 2000]
        B, NA, T = x.shape
        x = x.unsqueeze(1)  # [B, 1, 16, 2000]
        x = x.to(dtype=torch.float32)
        
        x = self.gelu(self.norm1(self.conv1(x)))  # [B, 5, 16, 400]
        x = F.interpolate(x, size=(16, 250), mode='nearest')
        x = self.gelu(self.norm2(self.conv2(x)))  # [B, 8, 16, 250]
        x = F.interpolate(x, size=(16, 200), mode='nearest')
        x = self.gelu(self.norm3(self.conv3(x)))  # [B, 10, 16, 200]
        
        x = self.gelu(self.norm4(self.conv4(x)))  # [B, 16, 16, 125]
        x = F.interpolate(x, size=(16, 100), mode='nearest')
        x = self.gelu(self.norm5(self.conv5(x)))  # [B, 20, 16, 100]
        x = F.interpolate(x, size=(16, 80), mode='nearest')
        x = self.gelu(self.norm6(self.conv6(x)))  # [B, 25, 16, 80]
        
        x = rearrange(x, 'B C NA T -> B NA (C T)')  # [B, 16, 2000]
        return x

class ChannelExpertMoE(nn.Module):
    def __init__(self, expert_output_dim=2000, hidden_dim=256):
        super().__init__()
        self.num_channels = 16
        self.time_dim = 2000
        self.temporal_conv = TemporalConv(in_chans=1)
    
    def forward(self, x): 
        time_feat = self.temporal_conv(x)  # [B, 16, 2000]
        return time_feat


class dw(nn.Module):
    def __init__(self, num_classes, expert_output_dim=2000, hidden_dim=256):
        super().__init__()
        self.moe = ChannelExpertMoE(expert_output_dim=expert_output_dim, hidden_dim=hidden_dim)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(expert_output_dim, num_classes)
        )
    def forward(self, x):
        moe_output = self.moe(x)
        features = moe_output.transpose(1, 2)

        logits = self.classifier(features)        
        return logits

if __name__ == '__main__':
    model = dw(num_classes=1)
    x = torch.randn(512, 16, 2000)
    output = model(x)
    print("output:", output.shape)  # Implementation note.
