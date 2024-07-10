from typing import Any

import numpy as np
from flax import linen as nn
from flax.training import train_state


class DataLoader(object):
    def __init__(self, x0, x1, rng=0, **kwargs):
        self.x0 = x0
        self.x1 = x1
        self.rng = np.random.default_rng(rng)

    def sample(self, batch_size=128, *args):
        idx = self.rng.choice(self.x0, size=(batch_size), replace=True)
        # idx_p = self.rng.choice(self.x1.shape[0], size=(batch_size), replace=True)
        return idx, idx


class TrainState(train_state.TrainState):
    batch_stats: Any


class Network(nn.Module):
    """A simple MLP classifier."""

    n_initial: int = 256
    n_hidden: int = 64
    n_layers: int = 3
    n_out: int = 1
    # act = nn.silu

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.n_initial)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.silu(x)
        for i in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.silu(x)
        x = nn.Dense(self.n_out)(x)
        return x


class ConditionalNetwork(nn.Module):
    """A simple MLP classifier."""

    n_initial: int = 256
    n_hidden: int = 64
    n_layers: int = 3
    n_out: int = 1
    # act = nn.silu

    @nn.compact
    def __call__(self, x, y):
        x = nn.concatenate([x, y], axis=-1)
        x = nn.Dense(self.n_initial)(x)
        x = nn.silu(x)
        for i in range(self.n_layers):
            # x = jnp.concatenate([x, y])
            x = nn.Dense(self.n_hidden)(x)
            x = nn.silu(x)
        x = nn.Dense(self.n_out)(x)
        return x


class TransformerNetwork(nn.Module):
    """A multihead attention transformer network."""

    n_heads: int = 4
    n_hidden: int = 64
    n_out: int = 1

    @nn.compact
    def __call__(self, x):
        # positional encoding
        n, d = x.shape
        pos = np.arange(d)
        pos = pos / np.power(10000, 2 * (pos // 2) / d)
        pos = pos[None, :]
        pos = np.stack([np.sin(pos), np.cos(pos)], axis=-1)
        x = x + pos
        # x = nn.Dense(self.n_hidden)(x)
        # x = nn.silu(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.n_hidden
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.n_out)(x)
        return x
