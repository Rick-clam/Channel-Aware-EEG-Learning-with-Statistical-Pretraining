import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F 
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint_sequential
import numpy as np

class TemporalConv(nn.Module):
    """EEG Temporal Conv with Extended Stages (No Linear Projection)"""
    def __init__(self, in_chans=1):
        super().__init__()
        # Implementation note.
        # Implementation note.
        
        # Implementation note.
        self.conv1 = nn.Conv2d(in_chans, 5, kernel_size=(1,15), stride=(1,5), padding=(0,7))  # padding/fill note.
        self.norm1 = nn.GroupNorm(1, 5)
        
        self.conv2 = nn.Conv2d(5, 8, kernel_size=(1,15), stride=(1,1), padding=(0,7))  # padding/fill note.
        self.norm2 = nn.GroupNorm(1, 8)
        
        self.conv3 = nn.Conv2d(8, 10, kernel_size=(1,15), stride=(1,1), padding=(0,7))  # padding/fill note.
        self.norm3 = nn.GroupNorm(1, 10)
        
        # Implementation note.
        self.conv4 = nn.Conv2d(10, 16, kernel_size=(1,3), stride=(1,2), padding=(0,1))  # padding/fill note.
        self.norm4 = nn.GroupNorm(1, 16)
        
        self.conv5 = nn.Conv2d(16, 20, kernel_size=(1,3), stride=(1,1), padding=(0,1))  # padding/fill note.
        self.norm5 = nn.GroupNorm(1, 20)
        
        self.conv6 = nn.Conv2d(20, 25, kernel_size=(1,3), stride=(1,1), padding=(0,1))  # padding/fill note.
        self.norm6 = nn.GroupNorm(1, 25)
        
        self.gelu = nn.GELU()

    def forward(self, x):  # [B,16,2000]
        B, NA, T = x.shape
        x = x.unsqueeze(1)  # [B,1,16,2000]

        x = x.to(dtype=torch.float32)  # Implementation note.
        # Implementation note.
        x = self.gelu(self.norm1(self.conv1(x)))  # Implementation note.
        x = torch.nn.functional.interpolate(x, size=(16,250), mode='nearest')  # Implementation note.
        x = self.gelu(self.norm2(self.conv2(x)))  # Implementation note.
        x = torch.nn.functional.interpolate(x, size=(16,200), mode='nearest')
        x = self.gelu(self.norm3(self.conv3(x)))  # Implementation note.
        
        # Implementation note.
        x = self.gelu(self.norm4(self.conv4(x)))  # Implementation note.
        x = torch.nn.functional.interpolate(x, size=(16,100), mode='nearest')
        x = self.gelu(self.norm5(self.conv5(x)))  # Implementation note.
        x = torch.nn.functional.interpolate(x, size=(16,80), mode='nearest')
        x = self.gelu(self.norm6(self.conv6(x)))  # Implementation note.
        
        # Implementation note.
        x = rearrange(x, 'B C NA T -> B NA (C T)')
        return x



class channelgate(nn.Module):
    """
ā?
: [B, C, T]
: [B, C, C]
    """
    def __init__(self, time_dim=2000, channel=16, embed_dim=64):
        super(channelgate, self).__init__()
        self.channel = channel

        # Implementation note.
        self.conv1d = nn.Conv1d(
            in_channels=channel,
            out_channels=embed_dim,
            kernel_size=9,
            stride=1,
            padding=4,
            groups=1
        )

        # Implementation note.
        self.project = nn.Sequential(
            nn.Linear(embed_dim, channel * channel),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        # Implementation note.
        diff = x[:, :, 1:] - x[:, :, :-1]  # [B, C, T-1]

        # Implementation note.
        feat = self.conv1d(diff)  # [B, embed_dim, T-1]

        # Implementation note.
        # Implementation note.
        feat_std = torch.std(feat, dim=2)  # [B, embed_dim]

        # Implementation note.
        weights = self.project(feat_std)  # [B, C*C]
        weights = weights.view(B, C, C)   # [B, C, C]
        return weights

class MLPExpert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x: [B*C, F_in]
        return self.net(x)

class MoE_EEG(nn.Module):
    def __init__(self, input_dim=2000, output_dim=100, num_channels=16, num_experts=4, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.num_experts = num_experts

        # Implementation note.
        self.experts = nn.ModuleList([
            MLPExpert(2000, output_dim, hidden_dim) for _ in range(num_experts)
        ])

        # Implementation note.
        self.shared_expert = MLPExpert(2000, output_dim, hidden_dim)

        # gating note.
        self.gate = nn.Sequential(
            nn.Linear(2000, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts + 1)
        )

    def forward(self, x):
        # print("x_flat:",x.shape)
        # x: [B, C, F_in]
        B, C, F_in = x.shape
        # print("C:",C)
        # print("F_in:",F_in)
        # assert C == self.num_channels and F_in == self.input_dim

        # reshape for gate input: [B*C, F_in]
        x_flat = x.reshape(B * C, F_in)
        # gate output: [B*C, num_experts+1]
        gate_logits = self.gate(x_flat)
        gate_weights = F.softmax(gate_logits, dim=-1)
        # gating note.

        # Implementation note.
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_flat))  # Implementation note.
        
        # Implementation note.
        expert_outputs.append(self.shared_expert(x_flat))  # Implementation note.

        # Implementation note.
        expert_stack = torch.stack(expert_outputs, dim=1)

        # gating note.
        gate_weights = gate_weights.unsqueeze(-1)  # [B*C, num_experts+1, 1]
        out = torch.sum(expert_stack * gate_weights, dim=1)  # [B*C, F_out]

        # Implementation note.
        out = out.view(B, C, self.output_dim)
        return out

class MeanOverChannels(nn.Module):
    def forward(self, x):
        # Implementation note.
        return x.mean(dim=1)  # Implementation note.

class ClassificationHead(nn.Sequential):
    def __init__(self,num_channels=16,num_classes=1):
        super().__init__()
        # Implementation note.
        self.clshead = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),     # [B, C, 1]
            nn.Flatten(),                # [B, C]
            nn.Linear(num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Implementation note.
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class LBTencoder(nn.Module):
    def __init__(self, num_channels=16, input_time=2000, compressed_time=100, num_classes=1):  # Implementation note.
        super().__init__()
        # Implementation note.
        self.temporal = TemporalConv()  # Implementation note.
        #self.compressed_time = compressed_time
        '''
        self.experts = nn.ModuleList([
            nn.Linear(input_time, compressed_time)
            for _ in range(num_channels)
        ])
        '''
       
        # gating note.
        self.gate = channelgate()
        
        '''
        self.classifier = nn.Sequential(
            MeanOverChannels(),  # Implementation note.
            nn.Linear(compressed_time, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Implementation note.
        )
        '''
        self.moe = MoE_EEG(input_dim=input_time, output_dim=compressed_time, num_channels=num_channels, num_experts=4, hidden_dim=256)
        #self.t = 0


    def extract_features(self, x):
        B, C, T = x.shape
        x_dwt = self.temporal(x)
        gate_weights = self.gate(x_dwt)
        fused_out = torch.einsum('bic,bij->bjc', x_dwt, gate_weights)
        #expert_out = self.moe(fused_out)         # [B, C, 100]
        pooled = fused_out.mean(dim=-1)  # Implementation note.
        return pooled  # [B, C]


    def forward(self, x, perturb = 0):  # Implementation note.
        # print("x:",x.shape)
        B, C, T = x.shape  
        # Implementation note.
        # if perturb:
        # Implementation note.
        # Implementation note.
        #     print("ts_new:",ts_new)
        # Implementation note.
        #     print("selected_ts:",selected_ts.shape)
        #     x = x[:,:,selected_ts]
        # TODO note.
        if perturb:
            ts = x.size(2)  # Implementation note.
            ts_new = np.random.randint(ts // 2, ts)  # Implementation note.
            selected_ts = np.random.choice(ts, ts_new, replace=False)  # Implementation note.
            
            # Implementation note.
            perturbed_x = torch.zeros_like(x)
            
            # Implementation note.
            # Implementation note.
            # global_mean = torch.mean(x)
            # perturbed_x.fill_(global_mean)
            
            # Implementation note.
            # channel_time_mean = torch.mean(x, dim=2, keepdim=True)  # [batch, 16, 1]
            # Implementation note.
            
            # Implementation note.
            noise_std = torch.std(x)
            perturbed_x = torch.randn_like(x) * noise_std
            
            # Implementation note.
            # Implementation note.
            
            # Implementation note.
            perturbed_x[:, :, selected_ts] = x[:, :, selected_ts]
            x = perturbed_x  # Implementation note.
            # print("perturbed_x:",perturbed_x.shape)

        # print("x1:",x.shape)
        
        # Implementation note.
        x_dwt = self.temporal(x)  # Implementation note.
        # Implementation note.


        # Implementation note.
        gate_weights = self.gate(x_dwt)  # [B, C, T] -> [B, C, C]
        #gate_weights = self.gate(expert_out)  # [B, C, T] -> [B, C, C]
        #print("expert_out:",expert_out.shape)
        #print("gate_weights:",gate_weights.shape)
        # Implementation note.
        #fused_out1 = torch.einsum('bic,bij->bjc', expert_out, gate_weights)  # [B, C, 100]
        fused_out = torch.einsum('bic,bij->bjc', x_dwt, gate_weights)  # [B, C, 2000]
        # print("fused_out:",fused_out.shape)
        
        #fused_out = self.temporal(fused_out)
        #fused_out = self.t*self.EEGChannelInteraction(expert_out) + (1-self.t)*fused_out1
        #fused_out = self.EEGChannelInteraction(expert_out)
        # expert_out = torch.cat([
        #     self.experts[i](fused_out[:, i, :]).unsqueeze(1)
        #     #self.experts[i](x[:,i,:]).unsqueeze(1)
        #     for i in range(C)
        # ], dim=1)  # [B, C, 100]

        # Implementation note.
        expert_out = self.moe(fused_out)   #[B,C,100]
        
        #expert_out = self.moe(x_dwt)
        # print("expert_out:",expert_out.shape)

        #out = self.classifier(x)
        #out = self.classifier(fused_out1)  # [B, num_classes]
        #out = self.classifier(fused_out)  # [B, num_classes]
        # out = self.classifier(expert_out)
        #out = self.classifier(gate_weights)
        # print("expert_out:",expert_out.shape)
        return expert_out

class EEGClassifierxr(nn.Module):
    def __init__(self, num_channels=16, input_time=2000, compressed_time=100, num_classes=1):  # Implementation note.
        super().__init__()
        self.lbt = LBTencoder(num_channels=num_channels, input_time=input_time, compressed_time=compressed_time, num_classes=num_classes)
        self.classifier = ClassificationHead(num_channels=num_channels,num_classes=num_classes)
    
    def forward(self, x):
        x = self.lbt(x)
        x = self.classifier(x)
        return x


# unsupervised pre-train module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, num_channels=16, input_time=2000, compressed_time=100, num_classes=1):
        super(UnsupervisedPretrain, self).__init__()
        self.lbt = LBTencoder(num_channels=num_channels, input_time=input_time, compressed_time=compressed_time, num_classes=num_classes)
        self.trans = nn.Linear(num_channels*compressed_time, 256, bias=True)  # Implementation note.
        self.prediction = nn.Sequential(  # Implementation note.
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        B, C, T = x.shape
        # print("UnsupervisedPretrain_x:",x.shape)
        emb = self.lbt(x,perturb=1)  # Implementation note.
        emb = emb.view(B, -1) #[512,1600]
        emb = self.trans(emb)
        # print("UnsupervisedPretrain_emb1:",x.shape)
        emb = self.prediction(emb)    
        # print("UnsupervisedPretrain_emb2:",emb.shape)   #[512,256]
        pred_emb = self.lbt(x)  # Implementation note.
        pred_emb  = pred_emb .view(B, -1)#[512,1600]
        pred_emb = self.trans(pred_emb)
        # print("UnsupervisedPretrain_pred_emb:",pred_emb.shape) #[512,256]
        return emb, pred_emb  # Implementation note.

# Implementation note.
if __name__ == '__main__':

    model = EEGClassifierxr(num_channels=16, input_time=2000, compressed_time=100, num_classes=1)
    
    x = torch.randn(512, 16, 2000)  # Implementation note.

    output = model(x)
    print("output:", output.shape)  # Implementation note.

    # model2 = channelgate(channel=16)
    # out = model2(x)
    # Implementation note.

    model = UnsupervisedPretrain(num_channels=16, input_time=2000, compressed_time=100, num_classes=1)
    out1, out2 = model(x)
    print("out1:",out1.shape)
    print("out2:",out2.shape)
