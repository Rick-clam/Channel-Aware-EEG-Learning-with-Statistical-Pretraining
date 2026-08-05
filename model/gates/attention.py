class AttentionGatedGate(nn.Module):
    def __init__(self, time_dim, num_experts, hidden_dim=256, num_heads=4):
        super().__init__()
        # Implementation note.
        self.time_attention = nn.MultiheadAttention(
            embed_dim=time_dim,  # Implementation note.
            num_heads=num_heads,
            batch_first=True
        )
        
        # Implementation note.
        self.feature_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU()
        )
        
        # Implementation note.
        self.weight_proj = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Implementation note.
        B, C, T = x.shape
        
        # Implementation note.
        x_flat = x.reshape(B*C, T, 1)  # Implementation note.
        attn_output, _ = self.time_attention(x_flat, x_flat, x_flat)  # Implementation note.
        
        # Implementation note.
        time_weights = torch.softmax(attn_output.squeeze(-1), dim=1)  # Implementation note.
        weighted_time_feat = (x.reshape(B*C, T) * time_weights).sum(dim=1)  # Implementation note.
        
        # Implementation note.
        feat = self.feature_proj(weighted_time_feat.unsqueeze(1))  # [B*C, hidden_dim]
        gate_weights = self.weight_proj(feat)  # [B*C, num_experts]
        return gate_weights.view(B, C, num_experts)  # [B, C, num_experts]
