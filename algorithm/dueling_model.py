import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # ===== Feature extractor =====
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # ===== Value stream =====
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # ===== Advantage stream =====
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        z = self.feature(x)

        value = self.value_stream(z)                 # (B, 1)
        advantage = self.advantage_stream(z)         # (B, A)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q