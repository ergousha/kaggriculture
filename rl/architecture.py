"""Neural Architecture for Kaggriculture Offline RL.

Defines the multi-modal Observation Encoder (Vectors + Spatial Grids)
and the Macro-Action Decoder for the Hybrid Strategy Policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEncoder(nn.Module):
    def __init__(self, vector_dim=4, spatial_channels=5, spatial_size=10, d_model=128):
        super().__init__()
        # Vector encoding (day, hour, money, hands)
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        
        # Spatial encoding (farm grid tiles: 5x10x10)
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 5x5
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Flattened size: 32 * 5 * 5 = 800
        spatial_out_dim = 32 * (spatial_size // 2) * (spatial_size // 2)
        self.spatial_proj = nn.Linear(spatial_out_dim, d_model)
        
    def forward(self, vector_obs, spatial_obs):
        v_emb = self.vector_mlp(vector_obs)
        s_emb = self.spatial_cnn(spatial_obs)
        s_emb = self.spatial_proj(s_emb)
        # Combine embeddings (additive fusion)
        return v_emb + s_emb

class MacroActionDecoder(nn.Module):
    def __init__(self, d_model=128, num_macro_actions=2):
        super().__init__()
        # Outputs logits for [did_hire, did_buy_land]
        self.macro_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_macro_actions)
        )
        
    def forward(self, state_emb):
        # We output raw logits for BCEWithLogitsLoss during training
        return self.macro_head(state_emb)

class KaggriculturePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ObservationEncoder()
        self.decoder = MacroActionDecoder()
        
    def forward(self, vector_obs, spatial_obs):
        state_emb = self.encoder(vector_obs, spatial_obs)
        return self.decoder(state_emb)

