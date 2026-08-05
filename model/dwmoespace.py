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

# ==============================================================================
# Implementation note.
# ==============================================================================

# Implementation note.
class EpilepsyChannelExpert(nn.Module):
    # Implementation note.
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.output_dim = output_dim
        self.high_freq_conv = nn.Conv1d(channel, embed_dim, kernel_size=3, stride=1, padding=1)
        self.sync_detector = nn.Sequential(nn.Linear(channel, channel), nn.Tanh())
        # Implementation note.
        self.relation_generator = nn.Linear(embed_dim + channel, channel * channel)
        self.feature_projection = nn.Linear(time_dim, output_dim)
        
    def forward(self, x):
        B, C, T = x.shape
        diff = x[:, :, 1:] - x[:, :, :-1]
        
        sync_feat = torch.mean(diff, dim=2)
        sync_feat = self.sync_detector(sync_feat)
        
        high_freq_feat = self.high_freq_conv(diff)
        high_freq_feat = torch.mean(high_freq_feat, dim=2)
        
        combined = torch.cat([high_freq_feat, sync_feat], dim=1) # [B, 63+16=79]
        relations = self.relation_generator(combined).view(B, C, C)
        return self.feature_projection(torch.matmul(relations, x))

class EpilepsyChannelExpert2(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim  # Implementation note.
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()  # Implementation note.
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()  # Implementation note.
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # Implementation note.

        # Implementation note.
        # Implementation note.
        diff = x[:, :, 1:] - x[:, :, :-1]  # [B, C, T-1]
        diff = F.pad(diff, (0, 1), mode='replicate')  # Implementation note.

        # Implementation note.
        encoded_feat = self.channel_encoder(diff)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)  # Implementation note.

        # Implementation note.
        channel_stats = x.mean(dim=2)  # Implementation note.
        channel_weights = self.channel_attn(channel_stats)  # Implementation note.

        # Implementation note.
        # Implementation note.
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)  # [B, embed_dim+C]
        # Implementation note.
        relations = self.relation_generator(combined_feat).view(B, C, C)  # [B, C, C]

        # Implementation note.
        # Implementation note.
        x_relation = torch.matmul(relations, x)  # [B, C, T]
        # Implementation note.
        x_weighted = x_relation * channel_weights.view(B, C, 1)  # [B, C, T]
        # Implementation note.
        x_out = x_weighted + x  # [B, C, T]

        return self.feature_projection(x_out)

# Implementation note.
class EmotionChannelExpert(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.output_dim = output_dim
        self.low_freq_conv = nn.Conv1d(channel, embed_dim, kernel_size=15, stride=1, padding=7)
        self.region_mapping = nn.Linear(channel, 4)
        # Implementation note.
        self.relation_generator = nn.Sequential(nn.Linear(embed_dim + 4, channel * channel), nn.Sigmoid())
        self.feature_projection = nn.Linear(time_dim, output_dim)
        
    def forward(self, x):
        B, C, T = x.shape
        smoothed = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        
        low_freq_feat = self.low_freq_conv(smoothed)
        low_freq_feat = torch.std(low_freq_feat, dim=2)
        
        channel_feat = torch.mean(smoothed, dim=2)
        region_feat = self.region_mapping(channel_feat)
        
        combined = torch.cat([low_freq_feat, region_feat], dim=1) # [B, 63+4=67]
        relations = self.relation_generator(combined).view(B, C, C)
        return self.feature_projection(torch.matmul(relations, x))

class EmotionChannelExpert2(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.output_dim = output_dim

        # Implementation note.
        self.low_freq_conv = nn.Conv1d(channel, embed_dim, kernel_size=15, stride=1, padding=7)
        self.low_freq_norm = nn.LayerNorm(embed_dim)
        
        # Implementation note.
        self.global_stat_proj = nn.Sequential(
            nn.Linear(channel * 2, embed_dim),  # Implementation note.
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Implementation note.
        # Implementation note.
        self.gate_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, channel * channel),
            nn.Sigmoid()
        )
        
        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)
        
    def forward(self, x):
        B, C, T = x.shape
        
        # Implementation note.
        # Implementation note.
        low_freq_feat = self.low_freq_conv(x)      # [B, embed_dim, T]
        low_freq_feat = low_freq_feat.mean(dim=2) # [B, embed_dim]
        low_freq_feat = self.low_freq_norm(low_freq_feat) # [B, embed_dim]

        # Implementation note.
        channel_mean = x.mean(dim=2)               # [B, C]
        channel_std = x.std(dim=2)                 # [B, C]
        global_stat_feat = self.global_stat_proj(torch.cat([channel_mean, channel_std], dim=1)) # [B, embed_dim]

        # Implementation note.
        # Implementation note.
        combined_features = torch.cat([low_freq_feat, global_stat_feat], dim=1) # [B, 2*embed_dim]
        # Implementation note.
        gate_weight = self.gate_fusion(combined_features) # [B, embed_dim]
        # Implementation note.
        fused_feat = (low_freq_feat * gate_weight) + (global_stat_feat * (1 - gate_weight)) # [B, embed_dim]

        # Implementation note.
        # Implementation note.
        relations = self.relation_generator(fused_feat).view(B, C, C) # [B, C, C]
        # Implementation note.
        x = torch.matmul(relations, x) + x # [B, C, T]

        # Implementation note.
        return self.feature_projection(x)



# Implementation note.
class WaveformChannelExpert2(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.output_dim = output_dim
        
        # Implementation note.
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(channel, embed_dim // 3, kernel_size=3, padding=1),
            nn.Conv1d(channel, embed_dim // 3, kernel_size=7, padding=3),
            nn.Conv1d(channel, embed_dim // 3, kernel_size=11, padding=5)
        ])
        
        # Implementation note.
        self.morphology_extractor = nn.Sequential(
            nn.Linear(channel * 2, channel),
            nn.GELU()
        )

        # Implementation note.
        self.channel_attention = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()  # Implementation note.
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Linear(embed_dim + channel, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, channel * channel),
            nn.Sigmoid()
        )
        
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape
        
        # Implementation note.
        scale_feats = [torch.mean(conv(x), dim=2) for conv in self.multi_scale_conv]  # [B, 21] * 3
        multi_scale_feat = torch.cat(scale_feats, dim=1)  # [B, 63]
        
        # Implementation note.
        peak_feat, valley_feat = torch.max(x, dim=2)[0], torch.min(x, dim=2)[0]  # [B, C] each
        morph_feat = self.morphology_extractor(torch.cat([peak_feat, valley_feat], dim=1))  # [B, C]
        
        # Implementation note.
        # Implementation note.
        channel_weights = self.channel_attention(morph_feat)  # [B, C]
        # Implementation note.
        x_weighted = x * channel_weights.view(B, C, 1)  # [B, C, T]
        
        # Implementation note.
        combined = torch.cat([multi_scale_feat, morph_feat], dim=1)  # [B, 63+16=79]
        relations = self.relation_generator(combined).view(B, C, C)  # [B, C, C]
        
        # Implementation note.
        x = torch.matmul(relations, x_weighted) + x  # [B, C, T]
        
        return self.feature_projection(x)

class TemporalDynamicsExpert(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.window_extractors = nn.ModuleList([
            nn.Conv1d(channel, embed_dim//3, kernel_size=10, stride=5, padding=5),  # padding/fill note.
            nn.Conv1d(channel, embed_dim//3, kernel_size=50, stride=10, padding=25),  # padding/fill note.
            nn.Conv1d(channel, embed_dim//3, kernel_size=100, stride=20, padding=50)  # padding/fill note.
        ])

        # Implementation note.
        self.dynamic_proj = nn.Conv1d(embed_dim, channel, kernel_size=1)

        # Implementation note.
        self.trend_encoder = nn.Sequential(
            nn.Linear(channel, channel * 2),
            nn.GELU(),
            nn.Linear(channel * 2, channel)  # Implementation note.
        )

        # Implementation note.
        self.consistency_attn = nn.Sequential(
            nn.Linear(embed_dim + channel, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, channel * channel),
            nn.Softmax(dim=-1)
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, 16, 2000]

        # Implementation note.
        window_feats = []
        for conv in self.window_extractors:
            feat = conv(x)  # Implementation note.
            # Implementation note.
            feat = F.interpolate(feat, size=T, mode='linear', align_corners=False)
            window_feats.append(feat)
        dynamic_feat = torch.cat(window_feats, dim=1)  # [B, 63, T]
        # Implementation note.
        dynamic_feat_proj = self.dynamic_proj(dynamic_feat)  # [B, 16, T]
        global_dynamic_feat = dynamic_feat.mean(dim=2)  # Implementation note.

        # Implementation note.
        # Implementation note.
        time_trend = torch.cumsum(x, dim=2) / (torch.arange(T, device=x.device) + 1)  # Implementation note.
        trend_feat = time_trend.mean(dim=2)  # [B, 16]
        trend_feat = self.trend_encoder(trend_feat)  # [B, 16]

        # Implementation note.
        combined = torch.cat([global_dynamic_feat, trend_feat], dim=1)  # [B, 63+16=79]
        consistency_matrix = self.consistency_attn(combined).view(B, C, C)  # [B, 16, 16]
        # Implementation note.
        x = torch.matmul(consistency_matrix, x)  # [B, 16, T]

        # Implementation note.
        x = x + dynamic_feat_proj  # Implementation note.
        return self.feature_projection(x)

class SpatialPatternExpert(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.num_groups = 4
        self.group_size = channel // self.num_groups  # Implementation note.
        assert self.num_groups * self.group_size == channel, "Channel count must be divisible by the number of groups."

        # Implementation note.
        self.group_extractor = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=1),  # Implementation note.
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

        # Implementation note.
        self.group_proj = nn.Conv1d(embed_dim, channel, kernel_size=1)

        # Implementation note.
        self.spatial_attn = nn.Sequential(
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.group_relation = nn.Sequential(
            nn.Linear(embed_dim + channel, self.num_groups * self.num_groups),
            nn.Softmax(dim=-1)
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [512, 16, 2000]

        # Implementation note.
        channel_stats = x.mean(dim=2)  # [B, 16]

        # Implementation note.
        group_feat = self.group_extractor(x)  # [B, 63, T]
        group_feat_proj = self.group_proj(group_feat)  # Implementation note.
        global_group_feat = group_feat.mean(dim=2)  # [B, 63]

        # Implementation note.
        spatial_weights = self.spatial_attn(channel_stats)  # [B, 16]
        x_weighted = x * spatial_weights.view(B, C, 1)  # [B, 16, T]

        # Implementation note.
        combined = torch.cat([global_group_feat, channel_stats], dim=1)  # [B, 79]
        group_rel_matrix = self.group_relation(combined).view(B, self.num_groups, self.num_groups)  # [B,4,4]
        
        # Implementation note.
        x_grouped = x_weighted.view(B, self.num_groups, self.group_size, T)  # [512,4,4,2000]
        
        # Implementation note.
        # Implementation note.
        x_grouped = torch.matmul(group_rel_matrix.unsqueeze(1), x_grouped.transpose(1,2)).transpose(1,2)
        
        # Implementation note.
        x_grouped = x_grouped.contiguous().view(B, C, T)  # [512,16,2000]

        # Implementation note.
        x = x_grouped + group_feat_proj  # Implementation note.
        return self.feature_projection(x)

class IrregularEventExpert(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.time_dim = time_dim  # Implementation note.

        # Implementation note.
        self.event_detector = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=5, padding=2, stride=1),  # padding/fill note.
            nn.BatchNorm1d(channel),
            nn.ReLU()
        )

        # Implementation note.
        self.event_encoder = nn.Sequential(
            nn.Linear(channel, embed_dim),
            nn.GELU()
        )

        # Implementation note.
        self.event_propagation = nn.Sequential(
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, 16, 2000]

        # Implementation note.
        kernel_size = 21
        # padding/fill note.
        local_mean = F.avg_pool1d(
            x, 
            kernel_size=kernel_size, 
            stride=1,  # Implementation note.
            padding=kernel_size // 2  # 21//2=10
        )  # Implementation note.

        # Implementation note.
        assert local_mean.shape[2] == T, f"Local mean time dimension mismatch: {local_mean.shape[2]} vs {T}"

        event_signal = torch.abs(x - local_mean)  # Implementation note.

        # Implementation note.
        event_feat = self.event_detector(event_signal)  # [B,16,2000]
        event_strength = event_feat.mean(dim=2)  # [B,16]
        event_duration = (event_feat > 0.1 * event_feat.max()).float().sum(dim=2)
        event_stats = event_strength * (event_duration / T)
        global_event_feat = self.event_encoder(event_stats)  # [B,63]

        # Implementation note.
        combined = torch.cat([global_event_feat, event_strength], dim=1)
        propagation_matrix = self.event_propagation(combined).view(B, C, C)
        x = torch.matmul(propagation_matrix, x)  # [B,16,2000]

        # Implementation note.
        x = x + event_feat  # Implementation note.
        return self.feature_projection(x)
# ==============================================================================
# Implementation note.
# ==============================================================================

# Implementation note.
class ChannelExpertMoE(nn.Module):
    def __init__(self, expert_output_dim=100, hidden_dim=256):
        super().__init__()
        self.num_channels = 16
        self.time_dim = 2000
        self.temporal_conv = TemporalConv(in_chans=1)
        self.experts = nn.ModuleList([
            EpilepsyChannelExpert2(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            EmotionChannelExpert2(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            WaveformChannelExpert2(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            # Implementation note.
            # MidFrequencyChannelExpert(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            # HighFrequencyChannelExpert(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),

            # Implementation note.
            # Implementation note.
                # channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            SpatialPatternExpert(  # Implementation note.
                channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            IrregularEventExpert(  # Implementation note.
                channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
        ])
        self.num_experts = len(self.experts)
        self.gate = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        time_feat = self.temporal_conv(x)  # [B, 16, 2000]
        B, C, F = time_feat.shape
        expert_outputs = [expert(time_feat) for expert in self.experts]
        x_flat = time_feat.reshape(B * C, F)
        gate_weights = self.gate(x_flat).view(B, C, self.num_experts)#[512, 16, 2]
        print("gate_weights1:",gate_weights.mean(dim=0).mean(dim=0))
        
        expert_stack = torch.stack(expert_outputs, dim=2) #[512, 16, 2, 100]
        # print("gate_weights2:",gate_weights.unsqueeze(-1).shape) #[512, 16, 2, 1]
        moe_output = torch.sum(expert_stack * gate_weights.unsqueeze(-1), dim=2) #[512, 16, 100]
        return moe_output



# ==============================================================================
# Implementation note.
# ==============================================================================
# Implementation note.
class dwmoespace(nn.Module):
    def __init__(self, num_classes, expert_output_dim=100, hidden_dim=256):
        super().__init__()
        self.moe = ChannelExpertMoE(expert_output_dim=expert_output_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(expert_output_dim, num_classes)
        )
        
    def forward(self, x):
        moe_output = self.moe(x)
        # print("moe_output:", moe_output.shape) #[512, 16, 100]
        features = moe_output.transpose(1, 2)
        # print("features:", features.shape) #[512, 100, 16]
        logits = self.classifier(features)
        # print("logits:", logits.shape) #[512, 6]
        return logits


# ==============================================================================
# Implementation note.
# ==============================================================================
if __name__ == '__main__':
    model = dwmoespace(num_classes=2)
    x = torch.randn(512, 16, 2000)
    output = model(x)
    print("output:", output.shape)  # Implementation note.
