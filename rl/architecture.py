"""Neural Architecture for Kaggriculture Full RL.

Defines the multi-modal Observation Encoder (Vectors + Spatial Grids)
and three distinct atomic action decoders:
1. FarmerHead (Categorical)
2. HandsSpatialHead (10x10 Spatial Categorical)
3. MarketHead (Multi-discrete transactions)
"""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(self, vector_dim=4, spatial_channels=5, spatial_size=10, d_model=128):
        super().__init__()
        # Vector encoding (day, hour, money, hands)
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 64), nn.ReLU(), nn.Linear(64, d_model)
        )

        # Spatial encoding (farm grid tiles: 5x10x10)
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 5x5
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Flattened size: 32 * 5 * 5 = 800
        spatial_out_dim = 32 * (spatial_size // 2) * (spatial_size // 2)
        self.spatial_proj = nn.Linear(spatial_out_dim, d_model)

    def forward(self, vector_obs, spatial_obs):
        v_emb = self.vector_mlp(vector_obs)
        s_emb = self.spatial_cnn(spatial_obs)
        s_emb = self.spatial_proj(s_emb)
        return v_emb + s_emb


class FarmerHead(nn.Module):
    def __init__(self, d_model=128, num_actions=20):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_actions))

    def forward(self, state_emb):
        return self.net(state_emb)


class HandsSpatialHead(nn.Module):
    def __init__(self, vector_dim=4, spatial_channels=5, num_actions=18):
        super().__init__()
        # We concatenate broadcasted vector features to the spatial grid
        in_channels = spatial_channels + vector_dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_actions, kernel_size=1),  # (B, num_actions, 10, 10)
        )

    def forward(self, vector_obs, spatial_obs):
        B, _, H, W = spatial_obs.shape
        # Broadcast vector obs
        v_grid = vector_obs.view(B, -1, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([spatial_obs, v_grid], dim=1)
        return self.conv_net(x)


class MarketHead(nn.Module):
    def __init__(self, d_model=128, num_items=12):
        super().__init__()
        self.num_items = num_items
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_items * 3),  # 3 channels: [buy, sell, log_qty]
        )

    def forward(self, state_emb):
        out = self.net(state_emb)
        return out.view(-1, self.num_items, 3)


class KaggriculturePolicyFullRL(nn.Module):
    def __init__(self, num_farmer_acts=20, num_hand_acts=18, num_market_items=12):
        super().__init__()
        self.encoder = ObservationEncoder()

        self.farmer_head = FarmerHead(num_actions=num_farmer_acts)
        self.hands_head = HandsSpatialHead(num_actions=num_hand_acts)
        self.market_head = MarketHead(num_items=num_market_items)

    def forward(self, vector_obs, spatial_obs):
        # Global state for Farmer and Market
        state_emb = self.encoder(vector_obs, spatial_obs)

        farmer_logits = self.farmer_head(state_emb)
        market_preds = self.market_head(state_emb)

        # Spatial grid for Hands
        hands_logits = self.hands_head(vector_obs, spatial_obs)

        return farmer_logits, hands_logits, market_preds
