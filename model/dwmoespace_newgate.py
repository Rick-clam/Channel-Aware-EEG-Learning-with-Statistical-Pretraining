import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==============================================================================
# Implementation note.
# ==============================================================================
class FilterAxisLayerNorm2d(nn.Module):
    """Normalize filter maps independently at every channel/time location.

    Unlike ``GroupNorm(1, C)``, this operation does not aggregate statistics
    across the temporal axis, so it preserves a finite temporal receptive field.
    """

    def __init__(self, num_filters, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_filters, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_filters, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (
            (x - mean)
            * torch.rsqrt(variance + self.eps)
            * self.weight
            + self.bias
        )


class TemporalConv(nn.Module):
    def __init__(self, in_chans=1, norm_type="legacy_global"):
        super().__init__()
        if norm_type not in ("legacy_global", "local_filter"):
            raise ValueError(
                "norm_type must be 'legacy_global' or 'local_filter'"
            )
        self.conv1 = nn.Conv2d(in_chans, 5, kernel_size=(1,15), stride=(1,5), padding=(0,7))
        self.conv2 = nn.Conv2d(5, 8, kernel_size=(1,15), stride=(1,1), padding=(0,7))
        self.conv3 = nn.Conv2d(8, 10, kernel_size=(1,15), stride=(1,1), padding=(0,7))
        self.conv4 = nn.Conv2d(10, 16, kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.conv5 = nn.Conv2d(16, 20, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv6 = nn.Conv2d(20, 25, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        norm = (
            (lambda channels: FilterAxisLayerNorm2d(channels))
            if norm_type == "local_filter"
            else (lambda channels: nn.GroupNorm(1, channels))
        )
        self.norm1 = norm(5)
        self.norm2 = norm(8)
        self.norm3 = norm(10)
        self.norm4 = norm(16)
        self.norm5 = norm(20)
        self.norm6 = norm(25)
        self.norm_type = norm_type
        self.gelu = nn.GELU()

    def forward(self, x, return_grid=False):  # [B, 16, 2000]
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
        
        if return_grid:
            # Preserve filter and physical latent-time axes.  The revised experts
            # convolve only over the final axis and therefore never cross a
            # filter-map boundary.
            return rearrange(x, 'B F C T -> B C F T')  # [B, 16, 25, 80]
        return rearrange(x, 'B F C T -> B C (F T)')  # [B, 16, 2000]



# class EpilepsyChannelExpert2(nn.Module):
#     def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
#         super().__init__()
#         self.channel = channel
# Implementation note.
#         self.output_dim = output_dim

# Implementation note.
#         self.channel_encoder = nn.Sequential(
#             nn.Conv1d(channel, embed_dim, kernel_size=3, stride=1, padding=1),
#             nn.GELU()
#         )

# Implementation note.
#         self.channel_attn = nn.Sequential(
#             nn.Linear(channel, channel // 2),
#             nn.GELU(),
#             nn.Linear(channel // 2, channel),
# Implementation note.
#         )

# Implementation note.
#         self.relation_generator = nn.Sequential(
#             nn.Linear(embed_dim + channel, channel * channel),
# Implementation note.
#         )

# Implementation note.
#         self.feature_projection = nn.Linear(time_dim, output_dim)

#     def forward(self, x):
# Implementation note.

# Implementation note.
# Implementation note.
#         diff = x[:, :, 1:] - x[:, :, :-1]  # [B, C, T-1]
# Implementation note.

# Implementation note.
#         encoded_feat = self.channel_encoder(diff)  # [B, embed_dim, T]
# Implementation note.

# Implementation note.
# Implementation note.
# Implementation note.

# Implementation note.
# Implementation note.
#         combined_feat = torch.cat([global_feat, channel_stats], dim=1)  # [B, embed_dim+C]
# Implementation note.
#         relations = self.relation_generator(combined_feat).view(B, C, C)  # [B, C, C]

# Implementation note.
# Implementation note.
#         x_relation = torch.matmul(relations, x)  # [B, C, T]
# Implementation note.
#         x_weighted = x_relation * channel_weights.view(B, C, 1)  # [B, C, T]
# Implementation note.
#         x_out = x_weighted + x  # [B, C, T]

#         return self.feature_projection(x_out)


# class EmotionChannelExpert2(nn.Module):
#     def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
#         super().__init__()
#         self.channel = channel
#         self.output_dim = output_dim

# Implementation note.
#         self.low_freq_conv = nn.Conv1d(channel, embed_dim, kernel_size=15, stride=1, padding=7)
#         self.low_freq_norm = nn.LayerNorm(embed_dim)
        
# Implementation note.
#         self.global_stat_proj = nn.Sequential(
# Implementation note.
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#         )
        
# Implementation note.
# Implementation note.
#         self.gate_fusion = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.Sigmoid()
#         )

# Implementation note.
#         self.relation_generator = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(embed_dim // 2, channel * channel),
#             nn.Sigmoid()
#         )
        
# Implementation note.
#         self.feature_projection = nn.Linear(time_dim, output_dim)
        
#     def forward(self, x):
#         B, C, T = x.shape
        
# Implementation note.
# Implementation note.
#         low_freq_feat = self.low_freq_conv(x)      # [B, embed_dim, T]
#         low_freq_feat = low_freq_feat.mean(dim=2) # [B, embed_dim]
#         low_freq_feat = self.low_freq_norm(low_freq_feat) # [B, embed_dim]

# Implementation note.
#         channel_mean = x.mean(dim=2)               # [B, C]
#         channel_std = x.std(dim=2)                 # [B, C]
#         global_stat_feat = self.global_stat_proj(torch.cat([channel_mean, channel_std], dim=1)) # [B, embed_dim]

# Implementation note.
# Implementation note.
#         combined_features = torch.cat([low_freq_feat, global_stat_feat], dim=1) # [B, 2*embed_dim]
# Implementation note.
#         gate_weight = self.gate_fusion(combined_features) # [B, embed_dim]
# Implementation note.
#         fused_feat = (low_freq_feat * gate_weight) + (global_stat_feat * (1 - gate_weight)) # [B, embed_dim]

# Implementation note.
# Implementation note.
#         relations = self.relation_generator(fused_feat).view(B, C, C) # [B, C, C]
# Implementation note.
#         x = torch.matmul(relations, x) + x # [B, C, T]

# Implementation note.
#         return self.feature_projection(x)



# Implementation note.
# class WaveformChannelExpert2(nn.Module):
#     def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
#         super().__init__()
#         self.channel = channel
#         self.output_dim = output_dim
        
# Implementation note.
#         self.multi_scale_conv = nn.ModuleList([
#             nn.Conv1d(channel, embed_dim // 3, kernel_size=3, padding=1),
#             nn.Conv1d(channel, embed_dim // 3, kernel_size=7, padding=3),
#             nn.Conv1d(channel, embed_dim // 3, kernel_size=11, padding=5)
#         ])
        
# Implementation note.
#         self.morphology_extractor = nn.Sequential(
#             nn.Linear(channel * 2, channel),
#             nn.GELU()
#         )

# Implementation note.
#         self.channel_attention = nn.Sequential(
#             nn.Linear(channel, channel // 2),
#             nn.GELU(),
#             nn.Linear(channel // 2, channel),
# Implementation note.
#         )

# Implementation note.
#         self.relation_generator = nn.Sequential(
#             nn.Linear(embed_dim + channel, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, channel * channel),
#             nn.Sigmoid()
#         )
        
#         self.feature_projection = nn.Linear(time_dim, output_dim)

#     def forward(self, x):
#         B, C, T = x.shape
        
# Implementation note.
#         scale_feats = [torch.mean(conv(x), dim=2) for conv in self.multi_scale_conv]  # [B, 21] * 3
#         multi_scale_feat = torch.cat(scale_feats, dim=1)  # [B, 63]
        
# Implementation note.
#         peak_feat, valley_feat = torch.max(x, dim=2)[0], torch.min(x, dim=2)[0]  # [B, C] each
#         morph_feat = self.morphology_extractor(torch.cat([peak_feat, valley_feat], dim=1))  # [B, C]
        
# Implementation note.
# Implementation note.
#         channel_weights = self.channel_attention(morph_feat)  # [B, C]
# Implementation note.
#         x_weighted = x * channel_weights.view(B, C, 1)  # [B, C, T]
        
# Implementation note.
#         combined = torch.cat([multi_scale_feat, morph_feat], dim=1)  # [B, 63+16=79]
#         relations = self.relation_generator(combined).view(B, C, C)  # [B, C, C]
        
# Implementation note.
#         x = torch.matmul(relations, x_weighted) + x  # [B, C, T]
        
#         return self.feature_projection(x)






# class IrregularEventExpert(nn.Module):
#     def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
#         super().__init__()
#         self.channel = channel
#         self.embed_dim = embed_dim
#         self.output_dim = output_dim
# Implementation note.

# Implementation note.
#         self.event_detector = nn.Sequential(
# padding/fill note.
#             nn.BatchNorm1d(channel),
#             nn.ReLU()
#         )

# Implementation note.
#         self.event_encoder = nn.Sequential(
#             nn.Linear(channel, embed_dim),
#             nn.GELU()
#         )

# Implementation note.
#         self.event_propagation = nn.Sequential(
#             nn.Linear(embed_dim + channel, channel * channel),
#             nn.Sigmoid()
#         )

# Implementation note.
#         self.feature_projection = nn.Linear(time_dim, output_dim)

#     def forward(self, x):
#         B, C, T = x.shape  # [B, 16, 2000]

# Implementation note.
#         kernel_size = 21
# padding/fill note.
#         local_mean = F.avg_pool1d(
#             x, 
#             kernel_size=kernel_size, 
# Implementation note.
#             padding=kernel_size // 2  # 21//2=10
# Implementation note.

# Implementation note.
# Implementation note.

# Implementation note.

# Implementation note.
#         event_feat = self.event_detector(event_signal)  # [B,16,2000]
#         event_strength = event_feat.mean(dim=2)  # [B,16]
#         event_duration = (event_feat > 0.1 * event_feat.max()).float().sum(dim=2)
#         event_stats = event_strength * (event_duration / T)
#         global_event_feat = self.event_encoder(event_stats)  # [B,63]

# Implementation note.
#         combined = torch.cat([global_event_feat, event_strength], dim=1)
#         propagation_matrix = self.event_propagation(combined).view(B, C, C)
#         x = torch.matmul(propagation_matrix, x)  # [B,16,2000]

# Implementation note.
# Implementation note.
#         return self.feature_projection(x)


# class EpilepsyChannelExpert2(nn.Module):
#     def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
#         super().__init__()
#         self.channel = channel
# Implementation note.
#         self.output_dim = output_dim

# Implementation note.
#         self.channel_encoder = nn.Sequential(
#             nn.Conv1d(channel, embed_dim, kernel_size=3, stride=1, padding=1),
#             nn.GELU()
#         )

# Implementation note.
#         self.channel_attn = nn.Sequential(
#             nn.Linear(channel, channel // 2),
#             nn.GELU(),
#             nn.Linear(channel // 2, channel),
# Implementation note.
#         )

# Implementation note.
#         self.relation_generator = nn.Sequential(
#             nn.Linear(embed_dim + channel, channel * channel),
# Implementation note.
#         )

# Implementation note.
#         self.feature_projection = nn.Linear(time_dim, output_dim)

#     def forward(self, x):
# Implementation note.

# Implementation note.
# Implementation note.
#         diff = x[:, :, 1:] - x[:, :, :-1]  # [B, C, T-1]
# Implementation note.

# Implementation note.
#         encoded_feat = self.channel_encoder(diff)  # [B, embed_dim, T]
# Implementation note.

# Implementation note.
# Implementation note.
# Implementation note.

# Implementation note.
# Implementation note.
#         combined_feat = torch.cat([global_feat, channel_stats], dim=1)  # [B, embed_dim+C]
# Implementation note.
#         relations = self.relation_generator(combined_feat).view(B, C, C)  # [B, C, C]

# Implementation note.
# Implementation note.
#         x_relation = torch.matmul(relations, x)  # [B, C, T]
# Implementation note.
#         x_weighted = x_relation * channel_weights.view(B, C, 1)  # [B, C, T]
# Implementation note.
#         x_out = x_weighted + x  # [B, C, T]

#         return self.feature_projection(x_out)

'''
####################################################################################
Implementation detail.
####################################################################################
'''
class EpilepsyChannelExpert_SmallScale(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=3, stride=1, padding=1),  # padding/fill note.
            nn.GELU()
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, C, T]

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)  # [B, embed_dim]

        # Implementation note.
        channel_stats = x.mean(dim=2)  # [B, C]
        channel_weights = self.channel_attn(channel_stats)  # [B, C]

        # Implementation note.
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)  # [B, embed_dim+C]
        relations = self.relation_generator(combined_feat).view(B, C, C)  # [B, C, C]

        # Implementation note.
        x_relation = torch.matmul(relations, x)  # Implementation note.
        x_weighted = x_relation * channel_weights.view(B, C, 1)  # Implementation note.
        x_out = x_weighted + x  # Implementation note.

        return self.feature_projection(x_out)

class EpilepsyChannelExpert_MediumScale(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=7, stride=1, padding=3),  # padding/fill note.
            nn.GELU()
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, C, T]

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)  # [B, embed_dim]

        # Implementation note.
        channel_stats = x.mean(dim=2)
        channel_weights = self.channel_attn(channel_stats)
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)
        relations = self.relation_generator(combined_feat).view(B, C, C)
        x_relation = torch.matmul(relations, x)
        x_weighted = x_relation * channel_weights.view(B, C, 1)
        x_out = x_weighted + x

        return self.feature_projection(x_out)

class EpilepsyChannelExpert_LargeScale(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=15, stride=1, padding=7),  # padding/fill note.
            nn.GELU()
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, C, T]

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)  # [B, embed_dim]

        # Implementation note.
        channel_stats = x.mean(dim=2)
        channel_weights = self.channel_attn(channel_stats)
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)
        relations = self.relation_generator(combined_feat).view(B, C, C)
        x_relation = torch.matmul(relations, x)
        x_weighted = x_relation * channel_weights.view(B, C, 1)
        x_out = x_weighted + x

        return self.feature_projection(x_out)


'''
####################################################################################
Implementation note.
####################################################################################
'''
class EpilepsyChannelExpert_SmallScale2(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100, dropout=0.3):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)  # Implementation note.
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Dropout(dropout),  # Implementation note.
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Dropout(dropout),  # Implementation note.
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, C, T]

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)  # [B, embed_dim]

        # Implementation note.
        channel_stats = x.mean(dim=2)  # [B, C]
        channel_weights = self.channel_attn(channel_stats)  # [B, C]

        # Implementation note.
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)  # [B, embed_dim+C]
        relations = self.relation_generator(combined_feat).view(B, C, C)  # [B, C, C]

        # Implementation note.
        x_relation = torch.matmul(relations, x)  # [B, C, T]
        x_weighted = x_relation * channel_weights.view(B, C, 1)  # [B, C, T]
        x_out = x_weighted + x  # Implementation note.

        return self.feature_projection(x_out)

class EpilepsyChannelExpert_MediumScale2(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100, dropout=0.3):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Dropout(dropout)  # Implementation note.
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, C, T]

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)  # [B, embed_dim]

        # Implementation note.
        channel_stats = x.mean(dim=2)  # [B, C]
        channel_weights = self.channel_attn(channel_stats)  # [B, C]

        # Implementation note.
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)  # [B, embed_dim+C]
        relations = self.relation_generator(combined_feat).view(B, C, C)  # [B, C, C]

        # Implementation note.
        x_relation = torch.matmul(relations, x)  # [B, C, T]
        x_weighted = x_relation * channel_weights.view(B, C, 1)  # [B, C, T]
        x_out = x_weighted + x  # Implementation note.

        return self.feature_projection(x_out)

class EpilepsyChannelExpert_LargeScale2(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100, dropout=0.3):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=15, stride=1, padding=7),
            nn.GELU(),
            nn.Dropout(dropout)  # Implementation note.
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.relation_generator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        # Implementation note.
        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape  # [B, C, T]

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)  # [B, embed_dim]

        # Implementation note.
        channel_stats = x.mean(dim=2)  # [B, C]
        channel_weights = self.channel_attn(channel_stats)  # [B, C]

        # Implementation note.
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)  # [B, embed_dim+C]
        relations = self.relation_generator(combined_feat).view(B, C, C)  # [B, C, C]

        # Implementation note.
        x_relation = torch.matmul(relations, x)  # [B, C, T]
        x_weighted = x_relation * channel_weights.view(B, C, 1)  # [B, C, T]
        x_out = x_weighted + x  # Implementation note.

        return self.feature_projection(x_out)

class EpilepsyChannelExpert_UltraSmallScale(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100, dropout=0.3):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=2, stride=1, padding=1),  # padding/fill note.
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        self.relation_generator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)

        # Implementation note.
        channel_stats = x.mean(dim=2)
        channel_weights = self.channel_attn(channel_stats)
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)
        relations = self.relation_generator(combined_feat).view(B, C, C)
        x_relation = torch.matmul(relations, x)
        x_weighted = x_relation * channel_weights.view(B, C, 1)
        x_out = x_weighted + x

        return self.feature_projection(x_out)

class EpilepsyChannelExpert_MediumLongScale(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100, dropout=0.3):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=11, stride=1, padding=5),  # padding/fill note.
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        self.relation_generator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)

        # Implementation note.
        channel_stats = x.mean(dim=2)
        channel_weights = self.channel_attn(channel_stats)
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)
        relations = self.relation_generator(combined_feat).view(B, C, C)
        x_relation = torch.matmul(relations, x)
        x_weighted = x_relation * channel_weights.view(B, C, 1)
        x_out = x_weighted + x

        return self.feature_projection(x_out)

class EpilepsyChannelExpert_UltraLargeScale(nn.Module):
    def __init__(self, channel=16, embed_dim=63, time_dim=2000, output_dim=100, dropout=0.3):
        super().__init__()
        self.channel = channel
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Implementation note.
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(channel, embed_dim, kernel_size=21, stride=1, padding=10),  # padding/fill note.
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Implementation note.
        self.channel_attn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        self.relation_generator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + channel, channel * channel),
            nn.Sigmoid()
        )

        self.feature_projection = nn.Linear(time_dim, output_dim)

    def forward(self, x):
        B, C, T = x.shape

        # Implementation note.
        encoded_feat = self.channel_encoder(x)  # [B, embed_dim, T]
        global_feat = encoded_feat.mean(dim=2)

        # Implementation note.
        channel_stats = x.mean(dim=2)
        channel_weights = self.channel_attn(channel_stats)
        combined_feat = torch.cat([global_feat, channel_stats], dim=1)
        relations = self.relation_generator(combined_feat).view(B, C, C)
        x_relation = torch.matmul(relations, x)
        x_weighted = x_relation * channel_weights.view(B, C, 1)
        x_out = x_weighted + x

        return self.feature_projection(x_out)
# ==============================================================================
# Implementation note.
# ==============================================================================

# Implementation note.
class TemporalChannelRelationExpert(nn.Module):
    """Channel-mixing expert whose scale kernel acts only on latent time.

    Input is ``[batch, EEG channel, filter map, latent time]``.  In contrast to
    the submitted implementation, filter maps are never concatenated into the
    convolution axis.
    """

    SUPPORTED_RELATIONS = (
        "none",
        "attention",
        "static",
        "dynamic",
        "dynamic_normalized",
        "spatial1x1",
        "static_conditioned_matched",
        "spatial1x1_conditioned_matched",
        "dynamic_normalized_attention",
    )

    def __init__(
        self,
        channel=16,
        filter_dim=25,
        latent_time=80,
        embed_dim=63,
        output_dim=100,
        temporal_kernel=7,
        relation_mode="dynamic_normalized",
        match_dynamic_budget=False,
    ):
        super().__init__()
        if temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd to preserve length")
        if relation_mode not in self.SUPPORTED_RELATIONS:
            raise ValueError(
                f"Unsupported relation_mode={relation_mode!r}; "
                f"choose from {self.SUPPORTED_RELATIONS}"
            )
        self.channel = channel
        self.filter_dim = filter_dim
        self.latent_time = latent_time
        self.relation_mode = relation_mode
        self.temporal_kernel = temporal_kernel

        # Conv2d treats filter maps as features and applies the kernel along the
        # final (latent-time) axis only. The EEG-channel spatial dimension has
        # kernel size one.
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(
                filter_dim,
                embed_dim,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
            ),
            nn.GELU(),
        )
        self.channel_attn = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.relation_generator = nn.Linear(embed_dim + channel, channel * channel)
        self.static_relation_logits = nn.Parameter(torch.zeros(channel, channel))
        self.spatial_conv = nn.Conv1d(channel, channel, kernel_size=1)
        self.feature_projection = nn.Linear(filter_dim * latent_time, output_dim)
        self.last_relation_matrix = None
        self.budget_adapter = None
        uses_conditioner = relation_mode.startswith("dynamic") or relation_mode in (
            "static_conditioned_matched",
            "spatial1x1_conditioned_matched",
        )
        if match_dynamic_budget and not uses_conditioner:
            # Approximately 33.5K parameters per expert (100.4K for three
            # experts), matching the trainable temporal-summary/relation
            # generators in the dynamic control within <0.1% total parameters.
            self.budget_adapter = nn.Sequential(
                nn.Linear(output_dim, 166),
                nn.GELU(),
                nn.Linear(166, output_dim),
            )
            nn.init.zeros_(self.budget_adapter[-1].weight)
            nn.init.zeros_(self.budget_adapter[-1].bias)

        uses_dynamic_relation = uses_conditioner
        uses_attention = relation_mode in (
            "attention",
            "dynamic_normalized_attention",
        )
        if not uses_dynamic_relation:
            for parameter in self.temporal_encoder.parameters():
                parameter.requires_grad_(False)
        if not uses_attention:
            for parameter in self.channel_attn.parameters():
                parameter.requires_grad_(False)
        if relation_mode not in ("static", "static_conditioned_matched"):
            self.static_relation_logits.requires_grad_(False)
        if relation_mode not in (
            "spatial1x1",
            "spatial1x1_conditioned_matched",
        ):
            for parameter in self.spatial_conv.parameters():
                parameter.requires_grad_(False)
        if not uses_conditioner:
            for parameter in self.relation_generator.parameters():
                parameter.requires_grad_(False)

    def forward(self, x, raw_channel_summary=None):
        if x.ndim != 4:
            raise ValueError(
                "True-temporal experts require [B,C,F,T] input; "
                f"received shape={tuple(x.shape)}"
            )
        batch_size, channels, filter_dim, latent_time = x.shape
        if (
            channels != self.channel
            or filter_dim != self.filter_dim
            or latent_time != self.latent_time
        ):
            raise ValueError(
                "Unexpected true-temporal feature grid: "
                f"expected [B,{self.channel},{self.filter_dim},{self.latent_time}], "
                f"received {tuple(x.shape)}"
            )

        if raw_channel_summary is None:
            flat_descriptors = x.reshape(
                batch_size, channels, filter_dim * latent_time
            )
            raw_channel_summary = torch.stack(
                (
                    flat_descriptors.mean(dim=-1),
                    flat_descriptors.std(dim=-1, unbiased=False),
                    flat_descriptors.abs().mean(dim=-1),
                ),
                dim=-1,
            )
        if raw_channel_summary.shape != (batch_size, channels, 3):
            raise ValueError(
                "raw_channel_summary must have shape "
                f"[B,{channels},3], received "
                f"{tuple(raw_channel_summary.shape)}"
            )
        # Channel standard deviation remains informative after the dataset
        # amplitude normalization and avoids the near-zero encoded means caused
        # by local filter-axis normalization.
        channel_stats = raw_channel_summary[..., 1]
        flat = x.reshape(batch_size, channels, filter_dim * latent_time)

        relation = None
        if self.relation_mode == "none":
            mixed = None
        elif self.relation_mode == "attention":
            weights = self.channel_attn(raw_channel_summary)
            mixed = flat * weights
        elif self.relation_mode == "static":
            relation = torch.softmax(self.static_relation_logits, dim=-1)
            relation = relation.unsqueeze(0).expand(batch_size, -1, -1)
            mixed = torch.bmm(relation, flat)
        elif self.relation_mode == "spatial1x1":
            mixed = self.spatial_conv(flat)
        elif self.relation_mode in (
            "static_conditioned_matched",
            "spatial1x1_conditioned_matched",
        ):
            encoded = self.temporal_encoder(x.permute(0, 2, 1, 3))
            global_feat = encoded.mean(dim=(2, 3))
            conditioner_logits = self.relation_generator(
                torch.cat([global_feat, channel_stats], dim=1)
            ).view(batch_size, channels, channels)
            channel_gain = 1.0 + torch.tanh(
                conditioner_logits.mean(dim=-1)
            ).unsqueeze(-1)
            if self.relation_mode == "static_conditioned_matched":
                relation = torch.softmax(
                    self.static_relation_logits, dim=-1
                ).unsqueeze(0).expand(batch_size, -1, -1)
                mixed = torch.bmm(relation, flat) * channel_gain
            else:
                mixed = self.spatial_conv(flat) * channel_gain
        else:
            encoded = self.temporal_encoder(x.permute(0, 2, 1, 3))
            global_feat = encoded.mean(dim=(2, 3))
            relation_logits = self.relation_generator(
                torch.cat([global_feat, channel_stats], dim=1)
            ).view(batch_size, channels, channels)
            if self.relation_mode == "dynamic":
                relation = torch.sigmoid(relation_logits)
            else:
                relation = torch.softmax(relation_logits, dim=-1)
            mixed = torch.bmm(relation, flat)
            if self.relation_mode == "dynamic_normalized_attention":
                mixed = mixed * self.channel_attn(raw_channel_summary)

        self.last_relation_matrix = (
            None if relation is None else relation.detach().cpu()
        )
        relation_residual = flat if mixed is None else mixed + flat
        output = self.feature_projection(relation_residual)
        if self.budget_adapter is not None:
            output = output + self.budget_adapter(output)
        return output


class ChannelExpertMoE(nn.Module):
    """Three-expert encoder with an explicitly selectable aggregation rule."""

    SUPPORTED_GATES = ("uniform", "static", "input")

    def __init__(
        self,
        expert_output_dim=100,
        hidden_dim=256,
        gate_type="static",
        gate_hidden_dim=16,
        expert_axis="legacy",
        relation_mode="dynamic_normalized",
        norm_type="auto",
        gate_descriptor_source="raw",
        match_dynamic_budget=False,
        expert_kernels=(3, 7, 15),
    ):
        super().__init__()
        self.num_channels = 16
        self.time_dim = 2000
        if gate_type not in self.SUPPORTED_GATES:
            raise ValueError(
                f"Unsupported gate_type={gate_type!r}; choose from {self.SUPPORTED_GATES}"
            )
        self.gate_type = gate_type
        if expert_axis not in ("legacy", "temporal"):
            raise ValueError("expert_axis must be 'legacy' or 'temporal'")
        self.expert_axis = expert_axis
        self.relation_mode = relation_mode
        if norm_type == "auto":
            norm_type = (
                "local_filter" if expert_axis == "temporal"
                else "legacy_global"
            )
        if norm_type not in ("legacy_global", "local_filter"):
            raise ValueError(
                "norm_type must be auto, legacy_global, or local_filter"
            )
        if gate_descriptor_source not in ("raw", "encoded"):
            raise ValueError(
                "gate_descriptor_source must be 'raw' or 'encoded'"
            )
        self.norm_type = norm_type
        self.gate_descriptor_source = gate_descriptor_source
        self.match_dynamic_budget = match_dynamic_budget
        if isinstance(expert_kernels, str):
            expert_kernels = tuple(
                int(value.strip())
                for value in expert_kernels.split(",")
                if value.strip()
            )
        else:
            expert_kernels = tuple(int(value) for value in expert_kernels)
        if not expert_kernels:
            raise ValueError("expert_kernels must contain at least one kernel")
        if any(kernel <= 0 or kernel % 2 != 1 for kernel in expert_kernels):
            raise ValueError("expert_kernels must contain positive odd integers")
        if expert_axis == "legacy" and expert_kernels != (3, 7, 15):
            raise ValueError(
                "custom expert_kernels are supported only for expert_axis='temporal'"
            )
        self.expert_kernels = expert_kernels
        self.temporal_conv = TemporalConv(
            in_chans=1, norm_type=norm_type
        )
        # TODO TODO TODO
        legacy_experts = [
            # Implementation note.
            # EpilepsyChannelExpert_SmallScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            # EpilepsyChannelExpert_MediumScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            # EpilepsyChannelExpert_LargeScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            # Implementation note.
            EpilepsyChannelExpert_SmallScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            EpilepsyChannelExpert_MediumScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            EpilepsyChannelExpert_LargeScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),

            # EpilepsyChannelExpert_UltraSmallScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            # EpilepsyChannelExpert_MediumLongScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
            # EpilepsyChannelExpert_UltraLargeScale(channel=self.num_channels, time_dim=self.time_dim, output_dim=expert_output_dim),
        ]
        temporal_experts = [
            TemporalChannelRelationExpert(
                channel=self.num_channels,
                output_dim=expert_output_dim,
                temporal_kernel=kernel,
                relation_mode=relation_mode,
                match_dynamic_budget=match_dynamic_budget,
            )
            for kernel in expert_kernels
        ]
        self.experts = nn.ModuleList(
            temporal_experts if expert_axis == "temporal" else legacy_experts
        )
        self.num_experts = len(self.experts)

        # Implementation note.
        # Implementation note.
        # TODO TODO TODO
        self.gate_weights = nn.Parameter(
            torch.zeros(self.num_channels, self.num_experts)
        )
        if self.gate_type == "uniform":
            self.gate_weights.requires_grad_(False)

        if self.gate_type == "input":
            self.input_gate = nn.Sequential(
                nn.Linear(3, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, self.num_experts),
            )
            nn.init.zeros_(self.input_gate[-1].weight)
            nn.init.zeros_(self.input_gate[-1].bias)
        else:
            self.input_gate = None
        self.last_gate_weights = None
        self.last_gate_descriptors = None

    def compute_gate_weights(self, time_feat):
        batch_size = time_feat.shape[0]
        if self.gate_type == "uniform":
            self.last_gate_descriptors = None
            return time_feat.new_full(
                (batch_size, self.num_channels, self.num_experts),
                1.0 / self.num_experts,
            )
        if self.gate_type == "static":
            self.last_gate_descriptors = None
            return torch.softmax(self.gate_weights, dim=-1).unsqueeze(0).expand(
                batch_size, -1, -1
            )
        flat_feat = time_feat.reshape(
            time_feat.shape[0], time_feat.shape[1], -1
        )
        channel_summary = torch.stack(
            (
                flat_feat.mean(dim=-1),
                flat_feat.std(dim=-1, unbiased=False),
                flat_feat.abs().mean(dim=-1),
            ),
            dim=-1,
        )
        self.last_gate_descriptors = channel_summary.detach().cpu()
        dynamic_residual = self.input_gate(channel_summary)
        return torch.softmax(
            self.gate_weights.unsqueeze(0) + dynamic_residual,
            dim=-1,
        )

    def forward(
        self, x, perturb=0, return_details=False
    ):  # Implementation note.
        # TODO note.
        if perturb:
            ts = x.size(2)  # Implementation note.
            ts_new = np.random.randint(ts // 2, ts)  # Implementation note.
            selected_ts = np.random.choice(ts, ts_new, replace=False)  # Implementation note.
            
            # Implementation note.
            perturbed_x = torch.zeros_like(x)
            
            # Implementation note.
            # Implementation note.
            global_mean = torch.mean(x)
            perturbed_x.fill_(global_mean)
            
            # Implementation note.
            # channel_time_mean = torch.mean(x, dim=2, keepdim=True)  # [batch, 16, 1]
            # Implementation note.
            
            # Implementation note.
            # noise_std = torch.std(x)
            # perturbed_x = torch.randn_like(x) * noise_std
            
            # Implementation note.
            # Implementation note.
            
            # Implementation note.
            perturbed_x[:, :, selected_ts] = x[:, :, selected_ts]
            x = perturbed_x  # Implementation note.
            # print("perturbed_x:",perturbed_x.shape)
        # time_feat = x
        time_feat = self.temporal_conv(
            x, return_grid=self.expert_axis == "temporal"
        )
        B, C = time_feat.shape[:2]
        raw_channel_summary = torch.stack(
            (
                x.mean(dim=-1),
                x.std(dim=-1, unbiased=False),
                x.abs().mean(dim=-1),
            ),
            dim=-1,
        )
        if self.expert_axis == "temporal":
            expert_outputs = [
                expert(
                    time_feat,
                    raw_channel_summary=raw_channel_summary,
                )
                for expert in self.experts
            ]
        else:
            expert_outputs = [expert(time_feat) for expert in self.experts]
        # print("self.experts_len:",expert_outputs)
        #===========================================================================
        # x_flat = time_feat.reshape(B * C, F)
        # # gate_weights = self.gate(x_flat).view(B, C, self.num_experts)#[512, 16, 2]
        # gate_weights = self.gate(time_feat).view(B, C, self.num_experts)#[512, 16, 2]
        # print("gate_weights1:",gate_weights.mean(dim=0).mean(dim=0))
        #=============================================================================

        # Implementation note.
        # Implementation note.
        gate_features = (
            x if self.gate_descriptor_source == "raw" else time_feat
        )
        gate_weights = self.compute_gate_weights(gate_features)
        self.last_gate_weights = gate_weights.detach().cpu()
        # print("gate_weights:",gate_weights.shape)
        # Implementation note.
        # if not perturb:
        #     print("gate_weights (mean across channels):", torch.softmax(self.gate_weights, dim=-1).mean(dim=0))
        expert_stack = torch.stack(expert_outputs, dim=2) #[512, 16, 2, 100]
        # print("expert_outputs:",len(expert_outputs))
        # print("expert_stack:",expert_stack.shape)
        # print("gate_weights2:",gate_weights.unsqueeze(-1).shape) #[512, 16, 2, 1]
        moe_output = torch.sum(expert_stack * gate_weights.unsqueeze(-1), dim=2) #[512, 16, 100]
        # print("moe_output:",moe_output.shape)
        if return_details:
            return moe_output, expert_stack, gate_weights
        return moe_output


class dwmoespace_newgate(nn.Module):
    def __init__(
        self,
        num_classes,
        expert_output_dim=100,
        hidden_dim=256,
        gate_type="static",
        gate_hidden_dim=16,
        expert_axis="legacy",
        relation_mode="dynamic_normalized",
        norm_type="auto",
        gate_descriptor_source="raw",
        match_dynamic_budget=False,
        expert_kernels=(3, 7, 15),
    ):
        super().__init__()
        self.moe = ChannelExpertMoE(
            expert_output_dim=expert_output_dim,
            hidden_dim=hidden_dim,
            gate_type=gate_type,
            gate_hidden_dim=gate_hidden_dim,
            expert_axis=expert_axis,
            relation_mode=relation_mode,
            norm_type=norm_type,
            gate_descriptor_source=gate_descriptor_source,
            match_dynamic_budget=match_dynamic_budget,
            expert_kernels=expert_kernels,
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(16,1),
        #     nn.Flatten(),
        #     nn.Linear(expert_output_dim, num_classes),
        #     nn.GELU(),
        # )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(expert_output_dim, num_classes)
        )
    def forward(self, x):
        moe_output = self.moe(x)
        # print("moe_output:", moe_output.shape) #[512, 16, 100]
        features = moe_output.transpose(1, 2)

        logits = self.classifier(features)        # print("features:", features.shape) #[512, 100, 16]
        # print("logits:", logits.shape) #[512, 6]
        return logits

    def forward_with_expert_details(self, x):
        """Return mixture and counterfactual expert outputs for held-out audits."""
        moe_output, expert_stack, gate_weights = self.moe(
            x, return_details=True
        )
        logits = self.classifier(moe_output.transpose(1, 2))
        expert_logits = torch.stack(
            [
                self.classifier(
                    expert_stack[:, :, expert_index, :].transpose(1, 2)
                )
                for expert_index in range(expert_stack.shape[2])
            ],
            dim=1,
        )
        classifier = self.classifier[-1]
        channel_expert_scores = F.linear(
            expert_stack,
            classifier.weight,
            bias=None,
        )
        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "gate_weights": gate_weights,
            "channel_expert_scores": channel_expert_scores,
        }


# Implementation note.
import numpy as np
class UnsupervisedPretrain(nn.Module):
    SUPPORTED_OBJECTIVES = (
        "unmasked_stats",
        "normalized_masked_stats",
        "shuffled_masked_stats",
        "masked_waveform",
    )

    def __init__(
        self,
        num_classes=1,
        expert_output_dim=100,
        hidden_dim=256,
        num_channels=16,
        compressed_time=100,
        need_token=20,
        raw_time=1000,
        gate_type="static",
        gate_hidden_dim=16,
        expert_axis="legacy",
        relation_mode="dynamic_normalized",
        norm_type="auto",
        gate_descriptor_source="raw",
        match_dynamic_budget=False,
        expert_kernels=(3, 7, 15),
        objective="unmasked_stats",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.raw_time = raw_time
        self.compressed_time = compressed_time  # Implementation note.
        self.need_token = need_token
        if objective not in self.SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unsupported objective={objective!r}; "
                f"choose from {self.SUPPORTED_OBJECTIVES}"
            )
        self.objective = objective
        if self.raw_time % need_token != 0:
            raise ValueError("raw_time must be divisible by need_token")
        self.raw_segment_len = self.raw_time // need_token
        
        # Implementation note.
        self.moe = ChannelExpertMoE(
            expert_output_dim=expert_output_dim,
            hidden_dim=hidden_dim,
            gate_type=gate_type,
            gate_hidden_dim=gate_hidden_dim,
            expert_axis=expert_axis,
            relation_mode=relation_mode,
            norm_type=norm_type,
            gate_descriptor_source=gate_descriptor_source,
            match_dynamic_budget=match_dynamic_budget,
            expert_kernels=expert_kernels,
        )
        
        # Implementation note.
        self.stats_features = [
            'mean',  # Implementation note.
            'std',  # Implementation note.
            # Implementation note.
            'skewness',  # Implementation note.
            # Implementation note.
            # Implementation note.
        ]
        self.num_stats = len(self.stats_features)  # Implementation note.
        
        if objective == "masked_waveform":
            # The same channel-shared decoder is applied to every channel. It
            # predicts the complete waveform, but loss is evaluated only on
            # segments that were hidden from the encoder.
            self.waveform_head = nn.Linear(expert_output_dim, raw_time)
            self.predict_heads = None
        else:
            # Each head predicts one statistic directly at the target segment
            # rate. It does not reconstruct a waveform-like intermediate.
            def build_pred_head():
                return nn.Sequential(
                    nn.Conv1d(
                        num_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                        groups=num_channels,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(need_token),
                )

            self.predict_heads = nn.ModuleList(
                [build_pred_head() for _ in range(self.num_stats)]
            )
            self.waveform_head = None

    def segment_input(self, x):
        B, C, T = x.shape
        if T != self.raw_time:
            raise ValueError(
                f"Expected {self.raw_time} samples, received input shape {tuple(x.shape)}"
            )
        return x.reshape(B, C, self.need_token, self.raw_segment_len)

    def compute_statistics(self, x):
        """Return clean mean/std/skew targets as [B,C,R,M]."""
        x_segmented = self.segment_input(x)
        raw_stats = []
        for stat in self.stats_features:
            if stat == 'mean':
                # Implementation note.
                stat  # Implementation note.
                stat_val = x_segmented.mean(dim=-1)  # [B,16,20]
            elif stat == 'std':
                # Implementation note.
                stat_val = x_segmented.std(dim=-1, unbiased=True)  # [B,16,20]
            elif stat == 'peak_to_peak':
                # Implementation note.
                stat_val = x_segmented.max(dim=-1).values - x_segmented.min(dim=-1).values  # [B,16,20]
            elif stat == 'skewness':
                # Implementation note.
                mean = x_segmented.mean(dim=-1, keepdim=True)
                std = x_segmented.std(dim=-1, unbiased=True, keepdim=True)
                # Implementation note.
                std = torch.where(std < 1e-6, torch.ones_like(std) * 1e-6, std)
                # Implementation note.
                stat_val = ((x_segmented - mean) / std).pow(3).mean(dim=-1)  # [B,16,20]
            elif stat == 'energy':
                # Implementation note.
                stat_val = (x_segmented ** 2).mean(dim=-1)  # [B,16,20]
            elif stat == 'kurtosis':
                # Implementation note.
                mean = x_segmented.mean(dim=-1, keepdim=True)
                std = x_segmented.std(dim=-1, unbiased=True, keepdim=True)
                std = torch.where(std < 1e-6, torch.ones_like(std) * 1e-6, std)
                # Implementation note.
                stat_val = ((x_segmented - mean) / std).pow(4).mean(dim=-1) - 3  # [B,16,20]
            raw_stats.append(stat_val)
        return torch.stack(raw_stats, dim=2)

    def forward(self, clean_x, encoder_x=None):
        """Predict clean targets from either complete or explicitly masked input."""
        if encoder_x is None:
            encoder_x = clean_x
        if encoder_x.shape != clean_x.shape:
            raise ValueError("clean_x and encoder_x must have identical shapes")
        moe_feat = self.moe(encoder_x)

        if self.objective == "masked_waveform":
            target = self.segment_input(clean_x)
            prediction = self.waveform_head(moe_feat).reshape_as(target)
            return target, prediction

        target = self.compute_statistics(clean_x)
        pred_stats = []
        for head in self.predict_heads:
            pred_stat = head(moe_feat)
            pred_stats.append(pred_stat)
        prediction = torch.stack(pred_stats, dim=2)
        return target, prediction

# Implementation note.

# ==============================================================================
# Implementation note.
# ==============================================================================
if __name__ == '__main__':
    model = dwmoespace_newgate(num_classes=1)
    x = torch.randn(512, 16, 2000)
    output = model(x)
    print("output:", output.shape)  # Implementation note.

    model = UnsupervisedPretrain(num_classes=1, raw_time=1000)
    out1, out2 = model(x[:, :, :1000])
    print("out1:",out1.shape)
    print("out2:",out2.shape)


