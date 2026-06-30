import random

import pytest
import torch


@pytest.fixture
def deterministic_seed():
    random.seed(0)
    torch.manual_seed(0)

    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_deterministic)
