"""Classification head + small auxiliary networks used inside the TQN neck."""
import torch
import torch.nn as nn


class Simple_CLS(nn.Module):
    """Final proficiency classifier applied on the fused ego/exo feature."""

    def __init__(self, in_dim, hidden_dim1=512, hidden_dim2=256, num_classes=4, dropout=0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class WeightMLP(nn.Module):
    """Per-token, per-branch (ego/exo) importance score used to gate the
    final ego-vs-exo fusion in `TQN.forward` (see models/neck.py)."""

    def __init__(self, input_dim=1024, hidden_dim=16):
        super().__init__()
        self.score_nets = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_mean):  # [T, B, C] -> [T, B, 1]
        return self.score_nets(feat_mean)


class WeightTransformer(nn.Module):
    """Per-view importance scorer used inside the exo-view fusion
    (TransformerDecoder in models/transformer.py) to redundancy-aware
    weight/select among the multiple exocentric views."""

    def __init__(self, input_dim=1024, num_heads=2, num_layers=1, num_views=3):
        super().__init__()
        dropout = 0.0
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, 1)  # per-view score

    def forward(self, x):
        encoded = self.encoder(x)  # [B, T, D]
        score_logits = self.output_layer(encoded)  # [B, T, 1]
        return score_logits


class VIBLayer(nn.Module):
    """Variational Information Bottleneck layer: reparameterized Gaussian
    bottleneck used for the ego-only / exo-only branches in the Gradient
    Blending (VIBGB) loss (see train.py)."""

    def __init__(self, input_dim, latent_dim, project_back=True):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.project_back = project_back

        if project_back:
            self.projector = nn.Sequential(
                nn.Linear(latent_dim, input_dim),
                nn.ReLU(),
            )

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization trick

        # KL(q(z|x) || N(0, I))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        if self.project_back:
            z = self.projector(z)

        return z, kl
