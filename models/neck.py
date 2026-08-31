"""
TQN (Transformer Query Network) neck: the core fusion module.

Pipeline:
  1. A set of learned query tokens attends (via TransformerDecoder, see
     transformer.py) into the ego memory sequence, and separately into the
     redundancy-aware fused exo memory (built from the N exocentric views
     inside TransformerDecoder when `exo=True`).
  2. Per-token ego/exo importance scores (`predictor_list`) are computed and
     turned into a soft ego-vs-exo gate via Gumbel-Softmax (train) / the raw
     scores (eval).
  3. The gated combination of ego and exo query features is the final fused
     feature fed to the classification head.
"""
import math
import torch
import torch.nn as nn
from .transformer import *
from .head import WeightMLP
import torch.nn.init as init


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model] -> broadcasts against [seq_len, batch, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [seq_len, batch_size, d_model]"""
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TQN(nn.Module):
    def __init__(self, d_model, num_queries, query_var=1, pe="no", H=4, Nego=2, Nexo=2,
                 feature_dim=1024, max_len=136, dropout=0.5):
        super(TQN, self).__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.query_var = query_var
        self.pe = pe
        self.max_len = max_len

        decoder_layer_ego = TransformerDecoderLayer(self.d_model, H, 1024, 0, 'relu', normalize_before=True)
        decoder_layer_exo = TransformerDecoderLayer(self.d_model, H, 1024, dropout, 'relu', normalize_before=True)

        decoder_norm_ego = nn.LayerNorm(self.d_model)
        decoder_norm_exo = nn.LayerNorm(self.d_model)
        self.decoder_ego = TransformerDecoder(decoder_layer_ego, Nego, decoder_norm_ego, return_intermediate=False)
        self.decoder_exo = TransformerDecoder(decoder_layer_exo, Nexo, decoder_norm_exo, return_intermediate=False)

        self.query_embed_ego = nn.Embedding(self.num_queries, self.d_model)
        self.query_embed_exo = nn.Embedding(self.num_queries, self.d_model)

        self.pos_encoder_ego = PositionalEncoding(self.d_model, max_len=self.max_len)
        self.pos_encoder_exo = PositionalEncoding(self.d_model, max_len=self.max_len)

        self.ego_proj = nn.Linear(self.d_model, self.d_model)
        self.exo_proj = nn.Linear(self.d_model, self.d_model)

        # ego/exo importance scorer, one small MLP per branch
        self.predictor_list = nn.ModuleList([
            WeightMLP(self.d_model) for _ in range(2)
        ])

    def forward(self, ego, exo_list, onehot, name=None, train=False, progressive=False, topk=2, tau=0.7):
        B = len(ego)

        query_embed_exo = self.query_embed_exo.weight.unsqueeze(1).repeat(1, B, 1)
        query_embed_ego = self.query_embed_ego.weight.unsqueeze(1).repeat(1, B, 1)

        if self.query_var != 1:
            query_embed_ego = init.normal_(query_embed_ego, mean=0, std=self.query_var)
            query_embed_exo = init.normal_(query_embed_exo, mean=0, std=self.query_var)

        pe_ego = self.pos_encoder_ego.pe[:query_embed_ego.size(0)].repeat(1, B, 1)
        query_pe_ego = self.pos_encoder_ego(query_embed_ego)

        pe_exo = self.pos_encoder_exo.pe[:query_embed_exo.size(0)].repeat(1, B, 1)
        query_pe_exo = self.pos_encoder_exo(query_embed_exo)

        ego_feat = ego.transpose(0, 1)
        exo_feats = [exo.transpose(0, 1) for exo in exo_list]

        # exo branch: internally fuses the N exo views (redundancy-aware
        # weighting, see TransformerDecoder(exo=True) in transformer.py)
        # before cross-attending the queries into them.
        features_exo, self_maps_exo, cross_maps_exo, memorys_exo, queries_exo, memory_ipt_exo, exoloss = \
            self.decoder_exo(query_embed_exo, exo_feats, memory_key_padding_mask=None,
                              pos=pe_exo, query_pos=query_pe_exo, train=train, exo=True, topk=topk, tau=tau)

        # ego branch: plain cross-attention into the single ego sequence.
        features_ego, self_maps_ego, cross_maps_ego, memorys_ego, queries_ego, memory_ipt_ego, _ = \
            self.decoder_ego(query_embed_ego, ego_feat, memory_key_padding_mask=None,
                              pos=pe_ego, query_pos=query_pe_ego, train=train, topk=topk)

        # per-token ego/exo importance scores -> gate for the final fusion
        ego_score = self.predictor_list[0](features_ego)  # [T, B, 1]
        exo_score = self.predictor_list[1](features_exo)  # [T, B, 1]
        scores = torch.cat([ego_score, exo_score], dim=-1)  # [T, B, 2]

        features = torch.stack([features_ego, features_exo], dim=2)  # [T, B, 2, C]

        if train:
            # soft, differentiable ego-vs-exo gate (Gumbel-Softmax), shared
            # across tokens (averaged score per sample)
            mean_scores = scores.mean(dim=0)
            soft_weights = F.gumbel_softmax(mean_scores, tau=tau, hard=False, dim=-1)  # [B, 2]
            soft_weights = soft_weights.unsqueeze(0).unsqueeze(-1).expand(features.size(0), -1, -1, -1)
            fused_feat = (features * soft_weights).sum(dim=2)
        else:
            # eval: use the raw per-token scores as soft weights directly
            soft_weights = scores.unsqueeze(-1)
            fused_feat = (features * soft_weights).sum(dim=2)

        return (fused_feat.permute(1, 0, 2), features_ego, features_exo,
                (self_maps_ego, cross_maps_ego, memorys_ego),
                [queries_ego, queries_exo], [memory_ipt_ego, memory_ipt_exo],
                scores, exoloss)
