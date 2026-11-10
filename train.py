"""Training and inference for the neural models."""
from __future__ import annotations

import copy
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config, FEATURES, TREND_FEATURES
from .data import SequenceSet
from .models import TPN, LSTMNet, TCNNet, GraphNet, NBeatsNet, DeepARNet


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def to_batch(S: SequenceSet, Xs: np.ndarray, idx=None, node_index: Optional[Dict[str, int]] = None,
             q_override: Optional[np.ndarray] = None):
    idx = np.arange(len(S)) if idx is None else np.asarray(idx)
    q = S.q if q_override is None else q_override
    b = {"x": torch.as_tensor(np.array(Xs[idx]), dtype=torch.float32),
         "q": torch.as_tensor(np.array(q[idx]), dtype=torch.float32),
         "tier": torch.as_tensor(np.array(S.tier[idx]), dtype=torch.long),
         "trend": torch.as_tensor(np.array(S.trend[idx]), dtype=torch.float32),
         "rel": torch.as_tensor(S.le_window[idx] - S.le_window[idx, -1:], dtype=torch.float32),
         "y": torch.as_tensor(np.array(S.y[idx]), dtype=torch.float32)}
    if node_index is not None:
        b["node"] = torch.as_tensor([node_index[c] for c in S.iso[idx]], dtype=torch.long)
    return b


def _batches(S: SequenceSet, cfg: Config, rng: np.random.RandomState, crosssection: bool):
    if crosssection:
        groups = [np.where(S.t0 == t)[0] for t in np.unique(S.t0)]
        rng.shuffle(groups)
        return [g for g in groups if len(g) >= 2]
    perm = rng.permutation(len(S))
    out = [perm[i:i + cfg.batch_size] for i in range(0, len(S), cfg.batch_size)]
    return [o for o in out if len(o) >= 2]


# ---------------------------------------------------------------------------
# Model factory (names used throughout the paper)
# ---------------------------------------------------------------------------
NEURAL_MODELS = ["LSTM", "TCN", "Graph (static)", "Graph (learned)", "N-BEATS", "DeepAR-style", "TPN"]

ABLATIONS = {
    "TPN (full)": {},
    "w/o trend skip": {"use_trend_skip": False},
    "w/o trend features": {"drop_features": [FEATURES.index(f) for f in TREND_FEATURES]},
    "w/o provenance attention bias": {"use_quality_bias": False},
    "w/o prototype": {"use_prototype": False},
    "w/o tier embedding": {"use_tier": False},
    "w/o SwiGLU (SiLU MLP)": {"use_swiglu": False},
    "w/o provenance loss weight": {"_no_qweight": True},
}


def make_model(name: str, cfg: Config, n_feat: int, n_nodes: int = 0, A_static=None, **tpn_kw):
    common = dict(n_feat=n_feat, horizon=cfg.horizon, hidden=cfg.hidden, dropout=cfg.dropout)
    if name == "TPN":
        kw = {k: v for k, v in tpn_kw.items() if not k.startswith("_")}
        return TPN(momentum=cfg.proto_momentum, **common, **kw)
    if name == "LSTM":
        return LSTMNet(**common)
    if name == "TCN":
        return TCNNet(**common)
    if name == "Graph (static)":
        return GraphNet(**common, n_nodes=n_nodes, adjacency="static", A_static=A_static)
    if name == "Graph (learned)":
        return GraphNet(**common, n_nodes=n_nodes, adjacency="learned")
    if name == "N-BEATS":
        return NBeatsNet(n_feat, cfg.horizon, seq_len=cfg.seq_len)
    if name == "DeepAR-style":
        return DeepARNet(**common)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------
def fit(model: nn.Module, S_tr: SequenceSet, X_tr, S_va: SequenceSet, X_va, cfg: Config, seed: int,
        node_index=None, quality_weight: bool = False, verbose: bool = False):
    """Train with early stopping on validation MAE (target space).
    The loss is computed on targets divided by the training-set standard
    deviation per horizon, so a single Huber delta is meaningful."""
    set_seed(seed)
    rng = np.random.RandomState(seed)
    cross = getattr(model, "needs_crosssection", False)
    gaussian = isinstance(model, DeepARNet)
    y_scale = torch.as_tensor(np.maximum(S_tr.y.std(0), 1e-4), dtype=torch.float32)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=10, factor=0.5, min_lr=1e-5)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(cfg.max_epochs):
        model.train()
        tot = 0.0
        for idx in _batches(S_tr, cfg, rng, cross):
            b = to_batch(S_tr, X_tr, idx, node_index)
            out = model(b)
            if gaussian:
                loss = 0.5 * (out["logvar"] + (b["y"] - out["mean"]) ** 2 / out["logvar"].exp())
                loss = loss.mean()
            else:
                err = (out["mean"] - b["y"]) / y_scale
                l = nn.functional.huber_loss(err, torch.zeros_like(err), delta=cfg.huber_delta,
                                             reduction="none").mean(1)
                if quality_weight:
                    w = 0.5 + ((b["q"] - 0.1) / 0.9).clamp(0, 1).mean(1)
                    l = l * w / w.mean()
                loss = l.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); tot += float(loss.detach())
        val_mae = float(np.abs(predict(model, S_va, X_va, node_index)["mean"] - S_va.y).mean())
        hist.append((ep, tot, val_mae))
        sched.step(val_mae)
        if val_mae < best - 1e-6:
            best, best_state, wait = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"    stopped at epoch {ep+1}, best val MAE {best:.4f}")
    return model, hist


@torch.no_grad()
def predict(model: nn.Module, S: SequenceSet, Xs, node_index=None, q_override=None) -> Dict[str, np.ndarray]:
    """Eval-mode inference. Graph models are evaluated one cross-section
    (forecast origin) at a time; all other models are per-sequence and
    independent of the other members of the batch."""
    model.eval()
    if getattr(model, "needs_crosssection", False):
        mean = np.zeros_like(S.y, dtype=float)
        for t in np.unique(S.t0):
            idx = np.where(S.t0 == t)[0]
            mean[idx] = model(to_batch(S, Xs, idx, node_index, q_override))["mean"].numpy()
        return {"mean": mean}
    out = model(to_batch(S, Xs, None, node_index, q_override))
    res = {"mean": out["mean"].numpy().astype(float)}
    if "logvar" in out:
        res["sd"] = np.exp(0.5 * out["logvar"].numpy().astype(float))
    if "attn" in out:
        res["attn"] = out["attn"].numpy()
    return res
