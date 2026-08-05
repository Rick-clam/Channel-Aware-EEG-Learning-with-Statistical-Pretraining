class TemporalSpatialGate(nn.Module):
    def __init__(self, num_channels, time_dim, num_experts, hidden_dim=256):
        super().__init__()
        self.num_channels = num_channels
        # Implementation note.
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10, stride=1, padding=4),  # padding/fill note.
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # Implementation note.
            nn.Flatten()  # Implementation note.
        )
        
        # Implementation note.
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1),  # padding/fill note.
            nn.GELU(),
            nn.AdaptiveAvgPool1d(time_dim)  # Implementation note.
        )
        
        # Implementation note.
        self.fusion = nn.Sequential(
            nn.Linear(8 + 16, hidden_dim),  # Implementation note.
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Implementation note.
        B, C, T = x.shape
        
        # Implementation note.
        x_temporal = x.reshape(B*C, 1, T)  # Implementation note.
        temporal_feat = self.temporal_encoder(x_temporal)  # [B*C, 8]
        
        # Implementation note.
        x_spatial = x  # [B, C, T]
        spatial_feat = self.spatial_encoder(x_spatial)  # [B, 16, T]
        spatial_feat = spatial_feat.mean(dim=2)  # Implementation note.
        spatial_feat = spatial_feat.unsqueeze(1).repeat(1, C, 1).reshape(B*C, 16)  # Implementation note.
        
        # Implementation note.
        fused = torch.cat([temporal_feat, spatial_feat], dim=1)  # [B*C, 24]
        gate_weights = self.fusion(fused)  # [B*C, num_experts]
        return gate_weights.view(B, C, num_experts)  # gating note.
