# src/models/tft.py
import math
from typing import Optional, Dict, Tuple
import torch
from torch import nn

# --------- Building blocks ---------
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.proj_gate = nn.Linear(d_out, 2 * d_out)  # for GLU
        self.glu = nn.GLU(dim=-1)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        y = self.fc1(x)
        y = self.elu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.glu(self.proj_gate(y))
        return self.norm(self.skip(x) + y)

class VariableSelectionNetwork(nn.Module):
    """
    Input x: (B, T, F). Projects each feature to d_model and uses learned softmax weights per time step.
    Returns (B, T, d_model) and weights (B, T, F, 1).
    """
    def __init__(self, num_features: int, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
        self.selector = GatedResidualNetwork(d_in=num_features, d_hidden=d_hidden, d_out=num_features, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, F = x.shape
        assert F == self.num_features
        parts = [self.proj[i](x[..., i:i+1]) for i in range(F)]      # F tensors of shape (B,T,d)
        H = torch.stack(parts, dim=2)                                 # (B,T,F,d)
        w = self.selector(x.reshape(B*T, F)).reshape(B, T, F, 1)      # (B,T,F,1)
        w = self.softmax(w)
        z = (H * w).sum(dim=2)                                        # (B,T,d)
        return z, w

class SimpleMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None):
        B, Tq, D = q.shape
        Tk = k.size(1)
        Q = self.q(q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        K = self.k(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        V = self.v(v).view(B, Tk, self.h, self.dk).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        A = torch.softmax(scores, dim=-1)
        A = self.drop(A)
        out = (A @ V).transpose(1, 2).contiguous().view(B, Tq, D)
        return self.o(out), A  # A: (B, h, Tq, Tk)

# --------- TFT Single-step ---------
class TFTSingleStep(nn.Module):
    """
    Inputs:
      past_inputs: (B, T, F)   (OHLCV + indicators + optional calendar features)
      static_id:   (B,) long   optional ticker ID (can be None)
    Returns:
      y_hat: (B,1), aux: {"vsn_weights": (B,T,F,1), "attn_weights": (B,h,1,T)}
    """
    def __init__(self,
                 num_features: int,
                 d_model: int = 128,
                 d_hidden: int = 256,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 static_num_ids: Optional[int] = None,
                 static_embed_dim: int = 16,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1):
        super().__init__()
        # Store init params for re-instantiating during model loading
        self._init_params = {
            "num_features": num_features,
            "d_model": d_model,
            "d_hidden": d_hidden,
            "n_heads": n_heads,
            "dropout": dropout,
            "static_num_ids": static_num_ids,
            "static_embed_dim": static_embed_dim,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers
        }

        self.static_num_ids = static_num_ids
        if static_num_ids is not None:
            self.id_embed = nn.Embedding(static_num_ids, static_embed_dim)
            self.static_fuse = GatedResidualNetwork(static_embed_dim, d_hidden, d_model, dropout)
        else:
            self.id_embed, self.static_fuse = None, None

        self.vsn = VariableSelectionNetwork(num_features, d_model, d_hidden, dropout)
        self.encoder = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden,
                               num_layers=lstm_layers, batch_first=True)
        self.attn = SimpleMHA(d_model=lstm_hidden, n_heads=n_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(lstm_hidden)

        dec_in = lstm_hidden + (d_model if self.static_fuse is not None else 0)
        self.decoder_grn = GatedResidualNetwork(dec_in, d_hidden, lstm_hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hidden, 1)
        )

    def forward(self, past_inputs: torch.Tensor, static_id: Optional[torch.Tensor] = None):
        B, T, F = past_inputs.shape
        aux: Dict[str, torch.Tensor] = {}

        z, w = self.vsn(past_inputs)           # (B,T,d), (B,T,F,1)
        aux["vsn_weights"] = w

        enc_out, _ = self.encoder(z)           # (B,T,H)
        q = enc_out[:, -1:, :]                 # (B,1,H)
        attn_out, A = self.attn(q, enc_out, enc_out)
        attn_out = self.attn_norm(attn_out + q).squeeze(1)  # (B,H)
        aux["attn_weights"] = A                # (B,h,1,T)

        if self.id_embed is not None and static_id is not None:
            if static_id.dim() == 2: static_id = static_id.squeeze(-1)
            s = self.id_embed(static_id)       # (B,E)
            s = self.static_fuse(s)            # (B,d)
            dec_in = torch.cat([attn_out, s], dim=-1)
        else:
            dec_in = attn_out

        dec = self.decoder_grn(dec_in)         # (B,H)
        y_hat = self.head(dec)                 # (B,1)
        return y_hat, aux
