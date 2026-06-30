# RankMixer

PyTorch implementation of RankMixer from ["RankMixer: Scaling Up Ranking Models in Industrial Recommenders"](https://arxiv.org/pdf/2507.15551).

This repository currently provides:

- A reusable RankMixer package interface.
- Core architecture classes (`RankMixer`, `RankMixerBlock`).
- Token-mixing and per-token dense/MoE layers.

## Current Status

The code is ready to import and run from source.
Packaging metadata, tests, and training scripts are being added in later steps.

## Install (Current Step)

Install runtime dependency:

```bash
pip install torch
```

Then run Python from the repository root so `rank_mixer` is importable.

## Minimal Usage

```python
import torch
from rank_mixer import RankMixer

model = RankMixer(
	d_model=128,
	token_dim=16,
	num_heads=16,
	num_layers=2,
	d_out=1,
	per_token_type='pffn',
)

x = torch.randn(32, 16, 128)   # (batch, tokens, dim)
logits = model(x)              # (32, 1)

print(logits.shape)
```

## Alternative Per-Token Layer (MoE)

```python
import torch
from rank_mixer import RankMixer

model = RankMixer(
	d_model=128,
	token_dim=16,
	num_heads=16,
	num_layers=2,
	d_out=1,
	per_token_type='premoe',
	num_experts=4,
)

x = torch.randn(8, 16, 128)
logits = model(x)
print(logits.shape)
```

## API Summary

### `RankMixer`

Constructor:

- `d_model`: feature dimension per token.
- `token_dim`: number of feature tokens.
- `num_heads`: token-mixing heads (needs to match token_dim).
- `num_layers`: number of stacked `RankMixerBlock` layers.
- `d_out`: prediction dimension.
- `per_token_type`: `'pffn'` or `'premoe'`.
- `ffn_expansion_ratio`: hidden expansion for FFN/MoE experts.
- `num_experts`: number of experts when using `'premoe'`.
- `eps`: LayerNorm epsilon.

Input and output:

- Input `x`: shape `(B, T, D)`.
- Output: shape `(B, d_out)`.

### `RankMixerBlock`

Reusable interaction block for plugging into larger architectures.

Block structure (pre-norm residual):

1. LayerNorm -> Token Mixing -> Residual add.
2. LayerNorm -> Per-token transformation (`PFFN` or `PReMoE`) -> Residual add.

Input and output shape are both `(B, T, D)`.

## Important Shape Constraint

The current token-mixing implementation enforces:

- `num_heads == token_dim`
- `d_model % num_heads == 0`

If these are not satisfied, initialization raises an error.

## Citation

If this implementation is useful, please cite the RankMixer paper.
