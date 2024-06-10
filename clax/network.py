from typing import Any

import numpy as np
import optax
from flax import linen as nn
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training import train_state


class DataLoader(object):
    def __init__(self, x0, x1, rng=0):
        self.x0 = np.atleast_2d(x0)
        self.x1 = np.atleast_2d(x1)
        self.rng = np.random.default_rng(rng)

    def sample(self, batch_size=128, *args):
        idx = self.rng.choice(self.x0.shape[0], size=(batch_size), replace=True)
        idx_p = self.rng.choice(self.x1.shape[0], size=(batch_size), replace=True)
        return idx, idx_p


class TrainState(train_state.TrainState):
    batch_stats: Any
    # scale_value: float = 1.0

    # def apply_gradients(self, *, grads, **kwargs):
    #     """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    #     Note that internally this function calls `.tx.update()` followed by a call
    #     to `optax.apply_updates()` to update `params` and `opt_state`.

    #     Args:
    #     grads: Gradients that have the same pytree structure as `.params`.
    #     **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    #     Returns:
    #     An updated instance of `self` with `step` incremented by one, `params`
    #     and `opt_state` updated by applying `grads`, and additional attributes
    #     replaced as specified by `kwargs`.
    #     """
    #     scale_value = kwargs.get("scale_value", 1.0)
    #     if OVERWRITE_WITH_GRADIENT in grads:
    #         grads_with_opt = grads["params"]
    #         params_with_opt = self.params["params"]
    #     else:
    #         grads_with_opt = grads
    #         params_with_opt = self.params

    #     updates, new_opt_state = self.tx.update(
    #         grads_with_opt, self.opt_state, params_with_opt, value=scale_value
    #     )
    #     new_params_with_opt = optax.apply_updates(params_with_opt, updates)

    #     # As implied by the OWG name, the gradients are used directly to update the
    #     # parameters.
    #     if OVERWRITE_WITH_GRADIENT in grads:
    #         new_params = {
    #             "params": new_params_with_opt,
    #             OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
    #         }
    #     else:
    #         new_params = new_params_with_opt
    #     return self.replace(
    #         step=self.step + 1,
    #         params=new_params,
    #         opt_state=new_opt_state,
    #         **kwargs,
    #     )

    # @classmethod
    # def create(cls, *, apply_fn, params, tx, **kwargs):
    #     """Creates a new instance with `step=0` and initialized `opt_state`."""
    #     # We exclude OWG params when present because they do not need opt states.
    #     params_with_opt = (
    #         params["params"] if OVERWRITE_WITH_GRADIENT in params else params
    #     )
    #     opt_state = tx.init(params_with_opt)
    #     return cls(
    #         step=0,
    #         apply_fn=apply_fn,
    #         params=params,
    #         tx=tx,
    #         opt_state=opt_state,
    #         **kwargs,
    #     )


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


from jax import numpy as jnp


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
