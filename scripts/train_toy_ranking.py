import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from rank_mixer import RankMixer


def sample_batch(batch_size: int, token_dim: int, d_model: int, device: torch.device):
    x = torch.randn(batch_size, token_dim, d_model, device=device)

    left = x[:, : token_dim // 2].mean(dim=(1, 2))
    right = x[:, token_dim // 2 :].mean(dim=(1, 2))
    y = (left > right).float().unsqueeze(-1)

    return x, y


def run_experiment(args: argparse.Namespace, per_token_type: str):
    device = torch.device(args.device)

    model = RankMixer(
        d_model=args.d_model,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_out=1,
        per_token_type=per_token_type,
        ffn_expansion_ratio=args.ffn_expansion_ratio,
        num_experts=args.num_experts,
    ).to(device)

    optim = AdamW(model.parameters(), lr=args.lr)

    print(f'\n[Run: {per_token_type}]')

    for step in range(1, args.steps + 1):
        model.train()

        x, y = sample_batch(args.batch_size, args.token_dim, args.d_model, device)

        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                preds = (logits.sigmoid() > 0.5).float()
                acc = (preds == y).float().mean().item()

            print(f'step={step:04d} loss={loss.item():.4f} acc={acc:.3f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train RankMixer on a toy ranking task')

    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log-every', type=int, default=50)

    parser.add_argument('--token-dim', type=int, default=16)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--ffn-expansion-ratio', type=int, default=4)
    parser.add_argument('--num-experts', type=int, default=4)

    parser.add_argument('--single-run', action='store_true', help='Run only pffn and skip premoe ablation')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    if args.token_dim % 2 != 0:
        raise ValueError('--token-dim must be even for the toy label rule')

    if args.num_heads != args.token_dim:
        raise ValueError('--num-heads must equal --token-dim for current token mixing implementation')

    if args.d_model % args.num_heads != 0:
        raise ValueError('--d-model must be divisible by --num-heads')

    return args


def main():
    args = parse_args()

    torch.manual_seed(42)

    run_experiment(args, per_token_type='pffn')

    if not args.single_run:
        run_experiment(args, per_token_type='premoe')


if __name__ == '__main__':
    main()