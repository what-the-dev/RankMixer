# Repo Concepts: fast-weight-attention -> RankMixer

## Goal

You want RankMixer to become a usable artifact, not just a single research implementation file.
The fastest path is to copy the repository product patterns from lucidrains/fast-weight-attention and adapt them to RankMixer.

This document breaks down:

1. What concepts make the lucidrains repo useful.
2. How each concept maps to RankMixer.
3. Concrete snippets you can drop in later when you decide to implement.

---

## What makes lucidrains/fast-weight-attention useful

### 1) Small, installable Python package with clean import surface

Pattern in lucidrains repo:

- Core logic lives inside a package directory.
- Package exposes a very small top-level API from __init__.py.
- User can do one-line import and run immediately.

Why this matters:

- Makes it pip-installable and reusable in any project.
- Reduces cognitive load for users.

Mapping to RankMixer:

- Move model code into a package folder rank_mixer/.
- Expose only primary public symbols (RankMixer, RankMixerBlock, maybe TokenMixer variants).

Useful snippet:

~~~python
# rank_mixer/__init__.py

from rank_mixer.model import RankMixer, RankMixerBlock
from rank_mixer.layers import MHTokenMixingBlock, PFFN, PReMoE

__all__ = [
		"RankMixer",
		"RankMixerBlock",
		"MHTokenMixingBlock",
		"PFFN",
		"PReMoE",
]
~~~

---

### 2) README as product entry point, not just a title

Pattern in lucidrains repo:

- README includes short concept statement, install command, minimal usage, and chunked usage variant.
- Usage examples are executable and shape-aware.

Why this matters:

- Users evaluate a repo in 30-60 seconds.
- Working examples are the strongest trust signal.

Mapping to RankMixer:

- README should include:
	- What RankMixer is and when to use it.
	- Install instructions.
	- Minimal forward pass example.
	- One practical recommendation task example (toy ranking objective).
	- API table for constructor args and expected input shape.

Useful snippet:

~~~python
import torch
from rank_mixer import RankMixer

model = RankMixer(
		d_model=128,
		token_dim=16,
		num_heads=16,
		num_layers=2,
		d_out=1,
		per_token_type="premoe",
		num_experts=4,
)

x = torch.randn(32, 16, 128)   # (batch, tokens, dim)
logits = model(x)              # (32, 1)
print(logits.shape)
~~~

---

### 3) Stateful or advanced behavior wrapped behind simple defaults

Pattern in lucidrains repo:

- Advanced features (memory carry, chunking, reverse-causal target, muon update) exist, but default call remains simple.
- Complexity is optional, not mandatory.

Why this matters:

- Beginners can use defaults.
- Advanced users can scale behavior without forking.

Mapping to RankMixer:

- Keep default RankMixer constructor easy.
- Hide experimental variants behind explicit flags or optional wrappers.
- Offer a convenience constructor for common recipe profiles.

Useful snippet:

~~~python
def rank_mixer_base(d_model=128, token_dim=16):
		return RankMixer(
				d_model=d_model,
				token_dim=token_dim,
				num_heads=token_dim,
				num_layers=2,
				per_token_type="pffn",
		)

def rank_mixer_moe(d_model=128, token_dim=16, num_experts=4):
		return RankMixer(
				d_model=d_model,
				token_dim=token_dim,
				num_heads=token_dim,
				num_layers=2,
				per_token_type="premoe",
				num_experts=num_experts,
		)
~~~

---

### 4) Toy training script that proves the idea

Pattern in lucidrains repo:

- train_toy.py shows end-to-end train loop, hyperparameters, and printed comparison conditions.
- Script is runnable with minimal setup.

Why this matters:

- Demonstrates model does something real.
- Makes repo educational and reproducible.

Mapping to RankMixer:

- Add a toy ranking task script where RankMixer must separate positive and negative candidates.
- Include at least one ablation toggle:
	- pffn vs premoe
	- token mixing on/off

Useful snippet:

~~~python
# scripts/train_toy_ranking.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from rank_mixer import RankMixer

def sample_batch(batch=64, tokens=16, dim=64):
		x = torch.randn(batch, tokens, dim)
		# Toy label: 1 if first half mean > second half mean
		y = (x[:, : tokens // 2].mean(dim=(1, 2)) > x[:, tokens // 2 :].mean(dim=(1, 2))).float()
		return x, y.unsqueeze(-1)

model = RankMixer(d_model=64, token_dim=16, num_heads=16, num_layers=2, d_out=1)
opt = AdamW(model.parameters(), lr=3e-4)

for step in range(1000):
		x, y = sample_batch()
		logits = model(x)
		loss = F.binary_cross_entropy_with_logits(logits, y)
		opt.zero_grad()
		loss.backward()
		opt.step()

		if step % 100 == 0:
				acc = ((logits.sigmoid() > 0.5) == y.bool()).float().mean().item()
				print(f"step={step} loss={loss.item():.4f} acc={acc:.3f}")
~~~

---

### 5) Tests for shape, API contracts, and regression safety

Pattern in lucidrains repo:

- Has tests directory.
- This signals reliability and catches breakages.

Why this matters:

- Others can safely depend on the repo.
- Refactors become less risky.

Mapping to RankMixer:

- Add pytest tests for:
	- output shapes
	- invalid shape raises
	- pffn and premoe both run
	- deterministic behavior under fixed seed (at least shape/value sanity)

Useful snippet:

~~~python
# tests/test_rank_mixer.py
import pytest
import torch
from rank_mixer import RankMixer

def test_forward_shape():
		model = RankMixer(d_model=32, token_dim=8, num_heads=8, num_layers=2, d_out=1)
		x = torch.randn(4, 8, 32)
		y = model(x)
		assert y.shape == (4, 1)

def test_bad_shape_raises():
		model = RankMixer(d_model=32, token_dim=8, num_heads=8)
		x = torch.randn(4, 7, 32)
		with pytest.raises(ValueError):
				_ = model(x)

@pytest.mark.parametrize("per_token_type", ["pffn", "premoe"])
def test_variants_run(per_token_type):
		model = RankMixer(
				d_model=32,
				token_dim=8,
				num_heads=8,
				per_token_type=per_token_type,
				num_experts=4,
		)
		x = torch.randn(2, 8, 32)
		y = model(x)
		assert y.shape == (2, 1)
~~~

---

### 6) Packaging metadata and dependency clarity

Pattern in lucidrains repo:

- pyproject.toml defines package metadata and dependencies.

Why this matters:

- Standard install path.
- Works with pip and modern tooling.

Mapping to RankMixer:

- Add pyproject.toml with:
	- package name and version
	- dependencies (torch minimum)
	- optional dev dependencies (pytest, ruff)

Useful snippet:

~~~toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rank-mixer"
version = "0.1.0"
description = "PyTorch implementation of RankMixer for ranking tasks"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
	"torch>=2.1",
]

[project.optional-dependencies]
dev = [
	"pytest>=8.0",
	"ruff>=0.5",
]
~~~

---

### 7) Reference implementation plus extension points

Pattern in lucidrains repo:

- Base implementation is compact and opinionated.
- Extra features are exposed as optional toggles.

Why this matters:

- Keeps core maintainable.
- Allows experimentation.

Mapping to RankMixer:

- Keep core path straightforward:
	- RankMixer
	- RankMixerBlock
	- Token mixer + per-token sublayer
- Add explicit extension points later:
	- alternative token mixers
	- sparse experts
	- sequence pooling heads

---

### 8) Research attribution and boundary clarity

Pattern in lucidrains repo:

- README cites related papers and implementation inspirations.

Why this matters:

- Clarifies what is original vs adapted.
- Helps users trust scientific grounding.

Mapping to RankMixer:

- Add concise citations section to README.
- Add a statement for implementation scope:
	- paper-faithful parts
	- known deviations/simplifications

Useful snippet:

~~~markdown
## Notes on Fidelity

This implementation aims to be paper-faithful for core RankMixer block structure
(token mixing + per-token transformation with pre-norm residual connections).
Some engineering choices are simplified for readability and educational use.
~~~

---

## Concrete concept mapping table

| fast-weight-attention concept | Why useful | RankMixer equivalent |
|---|---|---|
| Package directory + clean __init__ | Easy imports and reuse | rank_mixer/ package exposing RankMixer APIs |
| README with install and runnable usage | Fast onboarding | README with minimal inference + toy ranking training |
| Optional advanced features | Beginner-friendly + extensible | Keep defaults simple, add optional model profiles |
| train_toy.py | Demonstrates learning signal | scripts/train_toy_ranking.py with ablations |
| tests/ | Reliability | tests for shapes, variants, and error contracts |
| pyproject.toml | Installability and tooling | Standard metadata + deps + dev extras |
| Clear citations | Research context and trust | RankMixer paper citation + implementation notes |

---

## Suggested target repo shape for RankMixer

This is the practical structure to aim for:

~~~text
RankMixer/
	rank_mixer/
		__init__.py
		model.py
		layers.py
	scripts/
		train_toy_ranking.py
	tests/
		test_rank_mixer.py
	README.md
	pyproject.toml
	LICENSE
~~~

Your current state is good raw material, but not yet productized.
Most logic already exists in rank_mixer.py and can be split into model.py and layers.py with very little semantic change.

---

## What this means for your current rank_mixer.py

Current strengths:

- Clear class decomposition (RankMixer, RankMixerBlock, PFFN, PReMoE).
- Input validation and helpful shape errors.
- Configurable dense vs MoE per-token path.

Current blockers to external usability:

- No installable package structure.
- No runnable training or evaluation example.
- No tests.
- Very minimal README.
- No versioned packaging metadata.

---

## Recommended implementation order (when you are ready)

1. Package split and clean exports.
2. README minimal quickstart.
3. Toy training script.
4. Pyproject metadata.
5. Tests.
6. Optional polishing: examples, benchmark notes, ablation script.

This order gives the biggest usability jump early while keeping risk low.

---

## Optional API polish ideas

Useful additions inspired by lucidrains ergonomics:

- Add a method to count parameters by component (token mixer vs per-token layer).
- Add profile constructors (base, moe_small, moe_large).
- Add an ablation flag for disabling token mixing to compare behavior.
- Add small deterministic smoke test fixture.

Snippet example:

~~~python
def count_parameters(module):
		return sum(p.numel() for p in module.parameters() if p.requires_grad)

model = RankMixer(d_model=128, token_dim=16, num_heads=16, per_token_type="premoe", num_experts=4)
print("trainable params:", count_parameters(model))
~~~

---

## Final takeaway

The lucidrains repository is useful because it behaves like a small product:

- installable,
- readable,
- testable,
- runnable,
- and easy to start with.

RankMixer can reach the same bar quickly because the core modeling code already exists.
The highest leverage move is packaging + runnable examples + tests, in that order.
