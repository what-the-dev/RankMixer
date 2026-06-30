import torch

from rank_mixer import base, moe_large, moe_small


def _forward_once(constructor):
    model = constructor()
    x = torch.randn(2, model.token_dim, model.d_model)
    y = model(x)
    return y


def test_profile_constructors_forward_shapes(deterministic_seed):
    for constructor in (base, moe_small, moe_large):
        y = _forward_once(constructor)
        assert y.shape == (2, 1)


def test_profile_constructors_are_deterministic(deterministic_seed):
    torch.manual_seed(123)
    y1 = _forward_once(base)

    torch.manual_seed(123)
    y2 = _forward_once(base)

    assert torch.allclose(y1, y2)
