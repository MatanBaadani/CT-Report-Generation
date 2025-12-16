import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput

# =============================================================
# Utilities
# =============================================================

def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sinusoidal_position_1d(length: int, dim: int, device: torch.device):
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, dim)


# =============================================================
# Q-Former
# =============================================================

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=attn_dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=attn_dropout, batch_first=True)

        self.ln3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor, kv_mask: Optional[torch.Tensor] = None):

        residual = q_tokens
        qn = self.ln1(q_tokens)
        q_sa, _ = self.self_attn(qn, qn, qn, need_weights=False)
        q_tokens = residual + q_sa

        residual = q_tokens
        qn = self.ln2(q_tokens)
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask.bool()
        q_ca, _ = self.cross_attn(qn, kv_tokens, kv_tokens, key_padding_mask=key_padding_mask, need_weights=False)
        q_tokens = residual + q_ca

        residual = q_tokens
        qn = self.ln3(q_tokens)
        q_ffn = self.mlp(qn)
        q_tokens = residual + q_ffn
        return q_tokens


class QFormer(nn.Module):
    def __init__(self,
                 vision_dim: int,
                 hidden_dim: int = 512,
                 num_query_tokens: int = 128,
                 depth: int = 4,
                 n_heads: int = 8,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.1,
                 kv_posenc: Literal['none', 'sinusoidal', 'learned'] = 'none',
                 kv_pos_dim: Optional[int] = None,
                 max_kv_len: int = 8192):
        """
        A BLIP-2 style Q-Former that attends from learnable queries to frozen vision tokens.

        Args:
            vision_dim: channel size of your precomputed features (e.g., 768)
            hidden_dim: internal Q-Former hidden size
            num_query_tokens: number of learnable queries (64/128 are common)
            depth: number of cross-attention blocks
            kv_posenc: add positional encoding to KV tokens ('none'|'sinusoidal'|'learned')
            kv_pos_dim: PE dim (defaults to hidden_dim). If smaller, we'll right-pad; if larger, we'll project.
            max_kv_len: size of learned PE table if kv_posenc='learned'
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_query_tokens = num_query_tokens
        self.kv_posenc = kv_posenc
        self.kv_pos_dim = kv_pos_dim or hidden_dim


        self.proj_in = nn.Linear(vision_dim, hidden_dim)


        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim) * 0.02)


        if kv_posenc == 'learned':
            self.kv_pos_table = nn.Embedding(max_kv_len, self.kv_pos_dim)
        else:
            self.kv_pos_table = None

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio=4.0,
                                 attn_dropout=attn_dropout, proj_dropout=proj_dropout)
            for _ in range(depth)
        ])

        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)
        # If PE dim != hidden_dim and we use PE, keep a projector handy
        self._pe_proj: Optional[nn.Linear] = None

    def _add_kv_pos(self, kv: torch.Tensor) -> torch.Tensor:
        if self.kv_posenc == 'none':
            return kv
        B, N, H = kv.shape
        device = kv.device
        if self.kv_posenc == 'sinusoidal':
            pe = sinusoidal_position_1d(N, self.kv_pos_dim, device)  # (N, P)
        elif self.kv_posenc == 'learned':
            positions = torch.arange(N, device=device)
            pe = self.kv_pos_table(positions)  # (N, P)
        else:
            return kv

        if self.kv_pos_dim < H:
            pad = torch.zeros(N, H - self.kv_pos_dim, device=device)
            pe = torch.cat([pe, pad], dim=-1)
        elif self.kv_pos_dim > H:
            if self._pe_proj is None:
                self._pe_proj = nn.Linear(self.kv_pos_dim, H).to(device)
            pe = self._pe_proj(pe)

        pe = pe.unsqueeze(0).expand(B, N, H)
        return kv + pe

    def forward(self, vision_feats: torch.Tensor, vision_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, N, _ = vision_feats.shape
        kv = self.proj_in(vision_feats)
        kv = self._add_kv_pos(kv)
        kv = self.ln_kv(kv)

        q = self.query_tokens.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, kv_mask=vision_mask)
        return self.ln_out(q)


# =============================================================
# Q-Former → T5 Bridge (use Q tokens as the T5 encoder output)
# =============================================================

@dataclass
class BridgeConfig:
    t5_name: str = 't5-small'  # or 't5-base'
    freeze_t5_encoder: bool = True
    freeze_t5_decoder: bool = False  # often you fine-tune the decoder only


class QFormerT5(nn.Module):
    def __init__(self,
                 vision_dim: int = 768,
                 q_hidden_dim: int = 512,
                 num_query_tokens: int = 128,
                 q_depth: int = 4,
                 q_heads: int = 8,
                 kv_posenc: Literal['none', 'sinusoidal', 'learned'] = 'none',
                 kv_pos_dim: Optional[int] = None,
                 max_kv_len: int = 8192,
                 bridge_cfg: BridgeConfig = BridgeConfig()):
        super().__init__()

        # Q-Former
        self.qformer = QFormer(
            vision_dim=vision_dim,
            hidden_dim=q_hidden_dim,
            num_query_tokens=num_query_tokens,
            depth=q_depth,
            n_heads=q_heads,
            kv_posenc=kv_posenc,
            kv_pos_dim=kv_pos_dim,
            max_kv_len=max_kv_len,
        )

        # T5
        self.t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(bridge_cfg.t5_name)
        t5_d_model: int = self.t5.config.d_model

        self.q_to_t5 = nn.Linear(q_hidden_dim, t5_d_model)
                     
        if bridge_cfg.freeze_t5_encoder:
            for p in self.t5.encoder.parameters():
                p.requires_grad = False
        if bridge_cfg.freeze_t5_decoder:
            for p in self.t5.decoder.parameters():
                p.requires_grad = False
            for p in self.t5.lm_head.parameters():
                p.requires_grad = False

    def forward(self,
                vision_feats: torch.Tensor,           # (B, N, C_v)
                vision_mask: Optional[torch.Tensor] = None,  # (B, N) 1=valid
                decoder_input_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):

        q_tokens = self.qformer(vision_feats, vision_mask=vision_mask)  # (B, Q, H_q)
        enc_states = self.q_to_t5(q_tokens)  # (B, Q, d_model)


        encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)


        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=False,
        )
        return outputs

    @torch.no_grad()
    def generate(self,
                 vision_feats: torch.Tensor,
                 vision_mask: Optional[torch.Tensor] = None,
                 **gen_kwargs):
        q_tokens = self.qformer(vision_feats, vision_mask=vision_mask)
        enc_states = self.q_to_t5(q_tokens)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)
        return self.t5.generate(encoder_outputs=encoder_outputs, **gen_kwargs)


# # =============================================================
# # Example (dry run)
# # =============================================================
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Your saved encoder outputs after pooling+flatten
#     # Shape: (B, N, C_v) = (1, 4096, 768)
#     feats = torch.randn(1, 4096, 768, device=device)
#     attn_mask = torch.ones(1, 4096, dtype=torch.long, device=device)  # optional

#     model = QFormerT5(
#         vision_dim=768,           # matches your features
#         q_hidden_dim=512,         # internal Q-Former size
#         num_query_tokens=128,
#         q_depth=4,
#         q_heads=8,
#         kv_posenc='none',         # pooled tokens -> positions likely not meaningful
#         bridge_cfg=BridgeConfig(t5_name='t5-small', freeze_t5_encoder=True, freeze_t5_decoder=False)
#     ).to(device)

#     print(f"[PARAMS] total={sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
#           f"trainable={count_trainable_params(model)/1e6:.2f}M")

#     # Dry forward with labels (teacher forcing) to get a loss
#     tok = 32111  # <pad> for T5 vocab as a placeholder BOS; replace with real tokenizer usage
#     labels = torch.tensor([[tok, tok, tok]], device=device)
#     out = model(vision_feats=feats, vision_mask=attn_mask, labels=labels)
#     print("Loss:", float(out.loss))

#     # Generation demo (greedy, short)
#     ids = model.generate(vision_feats=feats, max_length=16)
#     print("Generated IDs shape:", ids.shape)
