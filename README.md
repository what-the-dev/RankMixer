# RankMixer

Unofficial PyTorch implementation of RankMixer from ["RankMixer: Scaling Up Ranking Models in Industrial Recommenders"](https://arxiv.org/pdf/2507.15551).

This repository provides:

- A reusable RankMixer package interface.
- Core architecture classes (`RankMixer`, `RankMixerBlock`).
- Token-mixing and per-token dense/MoE layers.
- Profile constructors for common model sizes (`base`, `moe_small`, `moe_large`).
- A toy training script for quick experimentation.
- Smoke tests for profile constructors.

## Install

Install from source (recommended):

```bash
pip install -e .
```

Install dev extras for tests/linting:

```bash
pip install -e .[dev]
```

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

## Profile Constructors

The package exports three convenience constructors:

- `base`: paper-faithful dense per-token FFN profile (`per_token_type='pffn'`).
- `moe_small`: smaller MoE profile (`per_token_type='premoe'`, fewer experts).
- `moe_large`: larger MoE profile (`per_token_type='premoe'`, more capacity).

### Quick Start with Profiles

```python
import torch
from rank_mixer import base, moe_small, moe_large

for constructor in (base, moe_small, moe_large):
	model = constructor()
	x = torch.randn(2, model.token_dim, model.d_model)
	y = model(x)
	print(constructor.__name__, y.shape)
```

Expected output shapes are `(2, 1)` for all three profiles.

### Profile Defaults

- `base(d_model=128, token_dim=16, num_layers=2, d_out=1, ffn_expansion_ratio=4, eps=1e-5)`
- `moe_small(d_model=128, token_dim=16, num_layers=2, d_out=1, ffn_expansion_ratio=4, num_experts=4, eps=1e-5)`
- `moe_large(d_model=192, token_dim=24, num_layers=3, d_out=1, ffn_expansion_ratio=4, num_experts=8, eps=1e-5)`

### Override Defaults

```python
from rank_mixer import moe_small

model = moe_small(
	d_model=256,
	token_dim=32,
	num_layers=4,
	num_experts=6,
)
```

When overriding values, keep the shape constraints described below.

## Alternative Per-Token Layer (Manual Constructor)

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

## Run Toy Training Script

Train on a synthetic ranking task:

```bash
python scripts/train_toy_ranking.py
```

Useful options:

```bash
python scripts/train_toy_ranking.py --steps 200 --log-every 20
python scripts/train_toy_ranking.py --single-run
```

The script runs `pffn` first and, unless `--single-run` is set, also runs a `premoe` ablation.

## Run Tests

```bash
pytest -q
```

## Citation

If this implementation is useful, please cite the RankMixer paper.
