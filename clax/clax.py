from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn
from jax import jit, tree_map
from tqdm import tqdm

from clax.network import DataLoader, Network, TrainState


@dataclass
class Trace:
    iteration: int = field(default=0)
    losses: list[float] = field(default_factory=list)


class Classifier(object):
    network: nn.Module = Network

    def __init__(self, n=2, **kwargs):
        """Initialise the network.

        args:
            n: int: Number of classes.
        kwargs:
            seed: int: Random seed.
        """
        self.rng = random.PRNGKey(kwargs.get("seed", 2024))
        self.network = Network(n_out=n)
        self.state = None

    def loss(self, params, batch, labels):
        """Loss function for training the calibrator."""
        output = self.state.apply_fn({"params": params}, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            output.squeeze(), labels
        ).mean()
        # loss = optax.sigmoid_binary_cross_entropy(
        #     output.squeeze(), labels
        # ).mean()
        return loss

    def _train(self, samples, labels, **kwargs):
        """Internal wrapping of training loop."""
        self.trace = Trace()
        labels = jnp.array(labels, dtype=int)
        samples = jnp.array(samples, dtype=jnp.float32)
        batch_size = kwargs.get("batch_size", 1024)
        data_size = samples.shape[0]
        epochs = kwargs.get("epochs", 10)
        epochs *= data_size // batch_size

        @jit
        def update_step(state, samples, labels):
            val, grads = jax.value_and_grad(self.loss)(state.params, samples, labels)
            state = state.apply_gradients(grads=grads, value=val)
            return val, state

        train_size = samples.shape[0]
        batch_size = min(batch_size, train_size)
        losses = []
        map = DataLoader(samples, labels)
        tepochs = tqdm(range(epochs))
        for k in tepochs:
            self.rng, step_rng = random.split(self.rng)
            perm, _ = map.sample(batch_size)
            batch = samples[perm, :]
            batch_label = labels[perm]
            loss, self.state = update_step(self.state, batch, batch_label)
            losses.append(loss)
            # self.state.losses.append(loss)
            if (k + 1) % 100 == 0:
                ma = jnp.mean(jnp.array(losses[-100:]))
                self.trace.losses.append(ma)
                tepochs.set_postfix(loss=ma)
                self.trace.iteration += 1

    def _init_state(self, **kwargs):
        dummy_x = jnp.zeros((1, self.ndims))
        _params = self.network.init(self.rng, dummy_x)
        lr = kwargs.get("lr", 1e-3)
        params = _params["params"]
        transition_steps = kwargs.get("transition_steps", 100)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr),
            optax.contrib.reduce_on_plateau(
                factor=0.5,
                patience=transition_steps,
                cooldown=transition_steps // 2,
            ),
        )

        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=optimizer,
        )

    def fit(self, samples, labels, **kwargs):
        """Calibrate the model on the provided data.

        Args:
            samples (np.ndarray): Samples to train on.
            labels (np.array): integer class labels corresponding to each sample.

        Keyword Args:
            restart (bool): If True, reinitialise the model before training. Defaults to False.
            batch_size (int): Size of the training batches. Defaults to 1024.
            epochs (int): Number of training epochs. Defaults to 10.
            lr (float): Learning rate. Defaults to 1e-3.
        """
        restart = kwargs.get("restart", False)
        self.ndims = samples.shape[-1]
        if (not self.state) | restart:
            self._init_state(**kwargs)
        self._train(samples, labels, **kwargs)
        self._predict_weight = lambda x: self.state.apply_fn(
            {
                "params": self.state.params,
            },
            x,
        )

    def predict(self, samples, log=True):
        """Predict the class (log) - probabilities for the provided samples.

        Args:
            samples (np.ndarray): Samples to predict on.
            log (bool): If True, return the log-probabilities. Defaults to True.
        """
        if log:
            return self._predict_weight(samples)
        else:
            return nn.softmax(self._predict_weight(samples))
