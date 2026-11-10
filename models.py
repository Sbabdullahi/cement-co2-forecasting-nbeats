"""Neural models. All take a batch dict and return {"mean": (B,H), ...} in the
target space y_h = log e_{t0+h} - log e_{t0}.

Batch keys: x (B,T,F scaled features), q (B,T raw provenance), tier (B,),
trend (B,H) trend baseline, rel (B,T) window-relative log emissions,
node (B,) country index (graph models only).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _qnorm(q: torch.Tensor) -> torch.Tensor:
    return ((q - 0.1) / 0.9).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Proposed model: quality-aware Temporal Prototype Network (TPN)
# ---------------------------------------------------------------------------
class TPN(nn.Module):
    """LSTM encoder -> provenance-biased additive attention pooling ->
    [pooled state ; scale-tier embedding ; cosine similarity to tier prototype]
    -> LayerNorm -> SwiGLU head -> residual added to the OLS trend baseline.

    Every component has a switch so that ablations remove exactly one thing.
    The prototype bank is a *buffer*: it is updated by an exponential moving
    average of training-batch embeddings and is read-only in eval mode, so
    validation/test data can never alter it (the original code updated it on
    every forward pass, including evaluation).
    """

    def __init__(self, n_feat: int, horizon: int, hidden: int = 32, n_tiers: int = 4,
                 dropout: float = 0.3, momentum: float = 0.9,
                 use_trend_skip: bool = True, use_quality_bias: bool = True,
                 use_prototype: bool = True, use_tier: bool = True, use_swiglu: bool = True,
                 drop_features: Optional[Sequence[int]] = None):
        super().__init__()
        self.cfg = dict(use_trend_skip=use_trend_skip, use_quality_bias=use_quality_bias,
                        use_prototype=use_prototype, use_tier=use_tier, use_swiglu=use_swiglu)
        self.momentum = momentum
        mask = torch.ones(n_feat)
        if drop_features:
            mask[list(drop_features)] = 0.0       # 0 == training mean after scaling
        self.register_buffer("feat_mask", mask)
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.attn_v = nn.Linear(hidden, 1, bias=False)
        self.attn_W = nn.Linear(hidden, hidden)
        if use_quality_bias:
            self.q_proj = nn.Linear(1, hidden, bias=False)
            self.q_beta = nn.Parameter(torch.tensor(1.0))
        if use_tier:
            self.tier_emb = nn.Embedding(n_tiers, 8)
        self.register_buffer("protos", torch.zeros(n_tiers, hidden))
        self.register_buffer("proto_seen", torch.zeros(n_tiers))
        d_in = hidden + (8 if use_tier else 0) + (1 if use_prototype else 0)
        self.norm = nn.LayerNorm(d_in)
        if use_swiglu:
            self.w_gate = nn.Linear(d_in, hidden)
            self.w_val = nn.Linear(d_in, hidden)
        else:
            self.w_in = nn.Linear(d_in, hidden)
        self.w_out = nn.Linear(hidden, horizon)

    @torch.no_grad()
    def _update_protos(self, z: torch.Tensor, tier: torch.Tensor):
        for k in tier.unique():
            zk = z[tier == k].mean(0)
            if self.proto_seen[k] == 0:
                self.protos[k] = zk
                self.proto_seen[k] = 1.0
            else:
                self.protos[k] = self.momentum * self.protos[k] + (1 - self.momentum) * zk

    def forward(self, b):
        x = b["x"] * self.feat_mask
        h, _ = self.lstm(x)
        h = self.drop(h)
        qn = _qnorm(b["q"])
        pre = self.attn_W(h)
        if self.cfg["use_quality_bias"]:
            pre = pre + self.q_proj(qn.unsqueeze(-1))
        s = self.attn_v(torch.tanh(pre)).squeeze(-1)
        if self.cfg["use_quality_bias"]:
            s = s + self.q_beta * qn
        a = torch.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        parts = [z]
        if self.cfg["use_tier"]:
            parts.append(self.tier_emb(b["tier"]))
        if self.cfg["use_prototype"]:
            if self.training:
                self._update_protos(z.detach(), b["tier"])
            seen = self.proto_seen[b["tier"]]
            sim = F.cosine_similarity(z, self.protos[b["tier"]], dim=-1, eps=1e-8) * seen
            parts.append(sim.unsqueeze(-1))
        f = self.norm(torch.cat(parts, dim=-1))
        if self.cfg["use_swiglu"]:
            u = F.silu(self.w_gate(f)) * self.w_val(f)
        else:
            u = F.silu(self.w_in(f))
        r = self.w_out(self.drop(u))
        mean = r + b["trend"] if self.cfg["use_trend_skip"] else r
        return {"mean": mean, "attn": a, "z": z}


# ---------------------------------------------------------------------------
# Deep baselines (same trend skip as TPN so the comparison isolates the encoder)
# ---------------------------------------------------------------------------
class LSTMNet(nn.Module):
    def __init__(self, n_feat, horizon, hidden=32, dropout=0.3, **_):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, b):
        _, (h, _) = self.lstm(b["x"])
        return {"mean": self.head(h[-1]) + b["trend"]}


class _CausalConv(nn.Module):
    def __init__(self, ci, co, k, d, dropout):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(ci, co, k, dilation=d)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        y = self.conv(F.pad(x, (self.pad, 0)))
        return F.gelu(self.drop(y)) + self.skip(x)


class TCNNet(nn.Module):
    """Causal dilated convolutions (left padding only), last time step read out."""

    def __init__(self, n_feat, horizon, hidden=32, dropout=0.3, **_):
        super().__init__()
        layers, ci = [], n_feat
        for i, co in enumerate([16, hidden, hidden]):
            layers.append(_CausalConv(ci, co, 3, 2 ** i, dropout)); ci = co
        self.enc = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, horizon))

    def forward(self, b):
        h = self.enc(b["x"].transpose(1, 2))[:, :, -1]
        return {"mean": self.head(h) + b["trend"]}


class GraphNet(nn.Module):
    """Cross-sectional graph model. A batch is the set of countries observed at
    one forecast origin t0, so message passing only mixes information that is
    available at t0. Adjacency is either a fixed country graph built from
    training-period data / external metadata ('static'), or learned from node
    embeddings with the MTGNN graph-learning layer (Wu et al., 2020; 'learned').
    """
    needs_crosssection = True

    def __init__(self, n_feat, horizon, n_nodes, hidden=32, dropout=0.3,
                 adjacency: str = "static", A_static: Optional[np.ndarray] = None,
                 emb_dim: int = 8, alpha: float = 3.0, **_):
        super().__init__()
        self.mode = adjacency
        self.enc = nn.GRU(n_feat, hidden, batch_first=True)
        self.g1 = nn.Linear(hidden, hidden)
        self.g2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)
        if adjacency == "static":
            A = torch.as_tensor(A_static if A_static is not None else np.eye(n_nodes), dtype=torch.float32)
            self.register_buffer("A", A)
        else:
            self.E1 = nn.Embedding(n_nodes, emb_dim)
            self.E2 = nn.Embedding(n_nodes, emb_dim)
            self.T1 = nn.Linear(emb_dim, emb_dim)
            self.T2 = nn.Linear(emb_dim, emb_dim)
            self.alpha = alpha

    def _adj(self, node):
        if self.mode == "static":
            A = self.A[node][:, node]
        else:
            M1 = torch.tanh(self.alpha * self.T1(self.E1(node)))
            M2 = torch.tanh(self.alpha * self.T2(self.E2(node)))
            A = F.relu(torch.tanh(self.alpha * (M1 @ M2.T - M2 @ M1.T)))
        A = A + torch.eye(len(node), device=A.device)
        return A / A.sum(1, keepdim=True)

    def forward(self, b):
        _, h = self.enc(b["x"]); h = h[-1]
        A = self._adj(b["node"])
        h1 = F.relu(self.g1(A @ h)) + h
        h2 = F.relu(self.g2(A @ self.drop(h1))) + h1
        return {"mean": self.head(self.drop(h2)) + b["trend"]}


class _NBeatsBlock(nn.Module):
    def __init__(self, T, H, hidden, kind, degree=2):
        super().__init__()
        self.kind = kind
        self.fc = nn.Sequential(nn.Linear(T, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        if kind == "trend":
            p = degree + 1
            tb = torch.arange(-T + 1, 1, dtype=torch.float32) / T
            tf = torch.arange(1, H + 1, dtype=torch.float32) / T
            self.register_buffer("Pb", torch.stack([tb ** i for i in range(p)]))
            self.register_buffer("Pf", torch.stack([tf ** i for i in range(p)]))
            self.theta_b = nn.Linear(hidden, p, bias=False)
            self.theta_f = nn.Linear(hidden, p, bias=False)
        else:
            self.back = nn.Linear(hidden, T)
            self.fore = nn.Linear(hidden, H)

    def forward(self, x):
        h = self.fc(x)
        if self.kind == "trend":
            return self.theta_b(h) @ self.Pb, self.theta_f(h) @ self.Pf
        return self.back(h), self.fore(h)


class NBeatsNet(nn.Module):
    """Compact N-BEATS (Oreshkin et al., 2020): a polynomial-trend stack whose
    trend coefficients are *learned*, followed by a generic stack, on the
    univariate window of relative log emissions. No fixed OLS baseline is
    added, which isolates 'learned decomposition' from 'fixed OLS skip'."""

    def __init__(self, n_feat, horizon, seq_len=8, hidden=64, **_):
        super().__init__()
        self.blocks = nn.ModuleList([_NBeatsBlock(seq_len, horizon, hidden, "trend"),
                                     _NBeatsBlock(seq_len, horizon, hidden, "trend"),
                                     _NBeatsBlock(seq_len, horizon, hidden, "generic"),
                                     _NBeatsBlock(seq_len, horizon, hidden, "generic")])

    def forward(self, b):
        res, fc = b["rel"], 0.0
        for blk in self.blocks:
            bc, f = blk(res)
            res = res - bc
            fc = fc + f
        return {"mean": fc}


class DeepARNet(nn.Module):
    """DeepAR-style probabilistic baseline (Salinas et al., 2020), simplified to
    direct multi-horizon Gaussian output (non-autoregressive) because H=3.
    Trained by Gaussian negative log-likelihood; outputs mean and log-variance."""

    def __init__(self, n_feat, horizon, hidden=32, dropout=0.3, **_):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mu = nn.Linear(hidden, horizon)
        self.lv = nn.Linear(hidden, horizon)

    def forward(self, b):
        _, (h, _) = self.lstm(b["x"])
        h = self.drop(h[-1])
        return {"mean": self.mu(h) + b["trend"], "logvar": self.lv(h).clamp(-12, 4)}
