class TaskAwareGate(nn.Module):
    def __init__(self, time_dim, num_channels, num_experts, hidden_dim=256):
        super().__init__()
        # Implementation note.
        self.rhythm_extractors = nn.ModuleList([
            # Implementation note.
            nn.Conv1d(1, 4, kernel_size=50, stride=1, padding=24),  # padding/fill note.
        ])
        
        # Implementation note.
        self.channel_prior = nn.Parameter(torch.randn(1, num_channels, 1))  # Implementation note.
        
        # Implementation note.
        self.fusion = nn.Sequential(
            nn.Linear(time_dim + 4, hidden_dim),  # Implementation note.
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        
        # Implementation note.
        x_rhythm = x.reshape(B*C, 1, T)  # [B*C, 1, T]
        rhythm_feat = self.rhythm_extractors[0](x_rhythm).mean(dim=2)  # Implementation note.
        
        # Implementation note.
        x_prior = x * self.channel_prior  # Implementation note.
        x_flat = x_prior.reshape(B*C, T)  # [B*C, T]
        
        # Implementation note.
        fused = torch.cat([x_flat, rhythm_feat], dim=1)  # [B*C, T+4]
        gate_weights = self.fusion(fused)  # [B*C, num_experts]
        return gate_weights.view(B, C, num_experts)
