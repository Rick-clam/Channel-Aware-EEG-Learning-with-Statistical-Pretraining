import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer
from einops import rearrange

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determining the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x):
        B, NA, T = x.shape
        #print("B, NA, T:",x.shape)
        x = x.unsqueeze(1)
        #print("x.shape",x.shape)
        x = self.gelu1(self.norm1(self.conv1(x)))
        #print("gelu1:",x.shape)
        x = self.gelu2(self.norm2(self.conv2(x)))
        #print("gelu2:",x.shape)
        x = self.gelu3(self.norm3(self.conv3(x)))
        #print("gelu3:",x.shape)
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        #print("rearrange:",x.shape)
        return x

class SpatialFilter(nn.Module):
    def __init__(self, in_emb, out_emb):
        super(SpatialFilter, self).__init__()
        # Implementation note.
        self.W = nn.Parameter(torch.randn(out_emb, in_emb))

    def forward(self, x):
        """
x: ㈢ (batch*channel, emb1)
: ㈢ (batch*channel, emb2)
        """
        return torch.einsum('bc,nc->bn', x, self.W)

class EEGFeatureExtractor(nn.Module):
    def __init__(self, n_fft=200, hop_length=50, win_length=50, use_log=True,
                 desired_dim=100, input_length=1000):
        """
        Implementation note.
n_fft: FFT у
hop_length: хЩ
            Implementation note.
Implementation detail.
Implementation detail.
input_length: ″︼?desired_dim ㄤ＄ STFT ?
        """
        super(EEGFeatureExtractor, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.use_log = use_log
        self.desired_dim = desired_dim
        self.activate = nn.GELU()
        

        # Implementation note.
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)
        
        # Implementation note.
        freq_bins = n_fft // 2 + 1
        #print("freq_bins:",freq_bins)
        time_frames = 1 + (input_length) // hop_length
        #print("time_frames:",time_frames)
        in_features = freq_bins * time_frames
        #print("in_features:",in_features)

        # Implementation note.
        if desired_dim is not None:
            self.fc = nn.Linear(in_features,desired_dim)
        else:
            self.fc = None

    def forward(self, x):
        """
        Implementation note.
x: tensor [batch, channels, time]
        Implementation note.
features: tensor [batch, channels, feature_out]
Implementation detail.
        """
        batch, channels, time_length = 128,16,x.shape[1]
        # Implementation note.
        
        # Implementation note.
        stft_output = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                                 win_length=self.win_length, window=self.window,
                                 return_complex=True)
        #print("stft_output:",stft_output.shape)

        # Implementation note.
        mag = stft_output.abs()
        if self.use_log:
            mag = torch.log1p(mag)
        
        # Implementation note.
        features = mag.flatten(start_dim=1)  # shape: [batch*channels, freq_bins * time_frames]

        # Implementation note.
        if self.fc is not None:
            features = self.fc(features)
        
        # Implementation note.
        #features = features.view(batch, channels, -1)
        features = self.activate(features)
        return features

class TopKFourierFilter(nn.Module):
    def __init__(self, feature_in, feature_out, top_k=30):
        super(TopKFourierFilter, self).__init__()
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.top_k = top_k

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, channel, feature_in]
        return: Tensor of shape [batch_size, channel, feature_out]
        """

        # Implementation note.
        x_fft = torch.fft.fft(x, dim=-1)  # [B, C, F]
        
        # Implementation note.
        magnitude = torch.abs(x_fft)  # [B, C, F]
        
        # Implementation note.
        # Implementation note.
        topk_vals, topk_indices = torch.topk(magnitude, self.top_k, dim=-1)
        
        # Implementation note.
        mask = torch.zeros_like(x_fft)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # Implementation note.
        x_fft_filtered = x_fft * mask
        
        # Implementation note.
        x_filtered = torch.fft.ifft(x_fft_filtered, dim=-1).real  # Implementation note.

        # Implementation note.
        if self.feature_out != self.feature_in:
            x_filtered = nn.functional.interpolate(
                x_filtered, size=self.feature_out, mode='linear', align_corners=False
            )

        return x_filtered

class LearnableFrequencyFilter(nn.Module):
    def __init__(self, feature_in, feature_out):
        super(LearnableFrequencyFilter, self).__init__()
        self.feature_in = feature_in
        self.feature_out = feature_out
        
        # Implementation note.
        self.freq_attention = nn.Sequential(
            nn.Linear(feature_in, feature_in),
            nn.ReLU(),
            nn.Linear(feature_in, feature_in),
            nn.Sigmoid()  # Implementation note.
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        
        # Step 1: FFT
        x_fft = torch.fft.fft(x, dim=-1)  # [B, C, F]
        
        # Implementation note.
        mag = torch.abs(x_fft)  # [B, C, F]
        mag_mean = mag.mean(dim=1)  # [B, F] or reduce to [B, 1, F] if needed
        
        freq_weights = self.freq_attention(mag_mean)  # [B, F]
        freq_weights = freq_weights.unsqueeze(1)  # [B, 1, F]

        # Implementation note.
        x_fft_filtered = x_fft * freq_weights  # [B, C, F]
        
        # Step 4: iFFT
        x_filtered = torch.fft.ifft(x_fft_filtered, dim=-1).real  # [B, C, T]
        
        # Implementation note.
        if self.feature_out != self.feature_in:
            x_filtered = nn.functional.interpolate(
                x_filtered, size=self.feature_out, mode='linear', align_corners=False
            )

        return x_filtered

class EEGChannelInteraction(nn.Module):  # Implementation note.
    def __init__(self, num_channels):
        super(EEGChannelInteraction, self).__init__()
        # Implementation note.
        self.channel_transform = nn.Linear(num_channels, num_channels, bias=True)

    def forward(self, x):
        # x: [batch_size, channel, time]
        # Implementation note.
        x = x.permute(0, 2, 1)  # Implementation note.
        combined = self.channel_transform(x)  # Implementation note.
        combined = combined.permute(0, 2, 1)  # Implementation note.
        return combined

class MLPExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.fc1 = nn.Linear(feature_in, feature_in // 2)
        self.fc2 = nn.Linear(feature_in // 2, feature_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class FourierExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.freq_dim = feature_in // 2 + 1  # Implementation note.
        self.fc = nn.Linear(self.freq_dim, feature_out)
        self.activation = nn.GELU()  # Implementation note.

    def forward(self, x):
        freq_x = torch.fft.rfft(x, dim=-1)  # Implementation note.
        freq_x = torch.abs(freq_x)  # Implementation note.
        x = self.fc(freq_x)  # Implementation note.
        x = self.activation(x)  # Implementation note.
        return x
       
'''
class ChannelRelationWeightPro(nn.Module):
    def __init__(self, in_feat=100, proj_dim=256, depth=4, n_heads=8, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.proj_dim = proj_dim

        # Implementation note.
        mlp_layers = []
        hidden_dim = proj_dim * 2
        mlp_layers.append(nn.LayerNorm(in_feat))
        mlp_layers.append(nn.Linear(in_feat, hidden_dim))
        mlp_layers.append(nn.GELU())
        for _ in range(depth - 1):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(hidden_dim, proj_dim))
        self.deep_mlp = nn.Sequential(*mlp_layers)

        # Implementation note.
        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=n_heads, batch_first=True)
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Implementation note.
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F)
        B, C, Fe = x.shape

        # Implementation note.
        feat = self.deep_mlp(x)                          # (B, C, D)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + self.eps)

        # Implementation note.
        feat_attn = self.attn_encoder(feat)              # (B, C, D)

        # Implementation note.
        R = torch.matmul(feat_attn, feat_attn.transpose(1, 2))  # (B, C, C)

        # Implementation note.
        I = torch.eye(C, device=x.device).unsqueeze(0)   # (1, C, C)
        R = (1 - self.gate) * R + self.gate * I

        # Implementation note.
        scores = R.sum(dim=-1) / self.temp               # (B, C)

        # Implementation note.
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, C, 1)

        # Implementation note.
        # Implementation note.
        fusion_factor = self.fusion_mlp(feat_attn).squeeze(-1)  # (B, C)
        weights = weights * fusion_factor.unsqueeze(-1)         # (B, C, 1)

        return weights
'''
class FourierMOE(nn.Module):  
    def __init__(self, feature_in, feature_out, expert_number,time_dim): #2000,100,4
        super().__init__()
        self.feature_out = feature_out
        self.EEGChannelInteraction = EEGChannelInteraction(num_channels=16)  # Implementation note.
        #self.MLPExpert = MLPExpert(feature_in,feature_out)
        #self.temporalconv = TemporalConv(in_chans=1,out_chans=8)

        self.experts = nn.ModuleList(
            [
               # EEGFeatureExtractor(feature_in, feature_out) for _ in range(expert_number)
                #EEGChannelInteraction(input_length=feature_in, desired_dim=feature_out) for _ in range(expert_number)
                FourierExpert(feature_in,feature_out) for _ in range(expert_number)
            ]
        )
        # gating note.
        #self.gate = SpatialFilter(feature_in, 1)
        self.gate = nn.Linear(feature_in, 1)  # gating note.
        #self.rel_mod = ChannelRelationWeightPro(in_feat=100)
        self.channel_groups = {  # Implementation note.
            0: [0, 1, 2, 3, 10, 11],
            1: [12, 13, 14, 15],
            2: [4, 5, 6, 7],
            3: [8, 9]
        }
        '''
        self.channel_groups = {  # Implementation note.
            0: [0, 1, 2, 3, 10, 11],
            1: [4, 5, 6, 7, 12, 13, 14, 15],
            2: [8, 9]
        }
        '''
        #self.TF = TopKFourierFilter(feature_in, feature_in, top_k=60)
        #self.LF = LearnableFrequencyFilter(feature_in, feature_in)
    def forward(self, x):
        # Implementation note.
        batch_size, channel, time = x.size()  #[512,16,2000]
        #print("batch_size:",batch_size)
        #print("channel:",channel)
        #print("time:",time)

        x = self.EEGChannelInteraction(x)

        # Implementation note.
        #gatex = self.positional_encoding(x)
        #print("gatex.shape:",gatex.shape)
        #print("gatextran.shape:",gatex.shape)
        # gating note.
        #x = self.temporalconv(x)
        #gatex = x.reshape(-1, time)
        #x = self.temporalconv(x)
        #x = self.TF(x)
        #x = self.LF(x)
        #x = self.temporalconv(x)
        
        expert_weight = self.gate(x)  # gating note.
        #print("expert_weight:",expert_weight.shape)
        # Implementation note.
        expert_weight = torch.softmax(expert_weight, dim=1)  # Implementation note.
        


        # Implementation note.

        # Implementation note.
        #['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
        # 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF']

        # Implementation note.
        # Implementation note.
        # Implementation note.
        # Implementation note.
        # Implementation note.

        outputs = [None] * channel  # Implementation note.
        for group_idx, channels in self.channel_groups.items():
            if not channels:
                continue
            # Collect data for the current group
            group_data = torch.cat([x[:, ch:ch+1, :] for ch in channels], dim=1)  # Implementation note.
            # Reshape to [batch * selected_channel, time]
            group_data = group_data.view(-1, time)
            # Apply expert
            group_output = self.experts[group_idx](group_data)  # Implementation note.
            #print("group_output:",group_output.shape)
            # Reshape back to [batch, channel, feature_out]
            feature_output = group_output.shape[-1]
            group_output = group_output.view(batch_size, len(channels), feature_output)  # Implementation note.
            # Assign outputs in the right positions
            for i, ch in enumerate(channels):
                outputs[ch] = group_output[:, i, :]  
        outputs = torch.stack(outputs, dim=1)  # Implementation note.
        # outputs [batchsize,channel,time] 512,16,100
        # print("output:",outputs.shape)
        # Apply expert weights and sum over channels
        
        #expert_weight = self.rel_mod(outputs)
        weighted_output = outputs * expert_weight
        summed_output = weighted_output.sum(dim=1)
        #print("summed_output:",summed_output.shape)

        return summed_output  # Implementation note.
        

class ClassificationHead(nn.Sequential):  # Implementation note.
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),  # Implementation note.
            nn.Linear(emb_size, n_classes),  # Implementation note.
            # Implementation note.
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class linear_MoEClassifier(nn.Module): 
    def __init__(self,batch_size, feature_in, feature_out, expert_number,n_classes):  #128,1000, 100,4,16,6
        super().__init__()
        self.MoE = FourierMOE(feature_in, feature_out, expert_number, feature_in) #1000, 100,4,16
        self.classifier = ClassificationHead(feature_out, n_classes)  #channel_number * feature_out

    def forward(self, x):
        #print("x.shape:",x.shape)
        x = self.MoE(x)  
        #print("x.moe:",x.shape)
        x = self.classifier(x)
        #print("x.classifier:",x.shape)
        return x



    # Implementation note.
    # Implementation note.
    #         input = sample.squeeze(1),
    # Implementation note.
    # Implementation note.
    #         center = False,
    # Implementation note.
    #         return_complex = True,
    #     )
    # Implementation note.
    #     return torch.abs(spectral)

    


if __name__ == "__main__":
    

    x = torch.randn(512,16,2560)

    model = linear_MoEClassifier(512,2560, 100,4,2)   
    out = model(x)
    print(out.shape)
    print(out)

