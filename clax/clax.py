from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn
from jax import jit, tree_map
from optax import tree_utils as otu
from tqdm import tqdm

from clax.network import DataLoader, Network, TrainState


@dataclass
class Trace:
    """
    Dataclass to store the training trace.
    """

    iteration: int = field(default=0)
    losses: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)


class Classifier(object):
    """Classifier class wrapping a basic jax multiclass classifier."""

    def __init__(self, n=2, **kwargs):
        """Initialise the Classifier.

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

    def _train(self, samples, labels, batches_per_epoch, **kwargs):
        """Internal wrapping of training loop."""
        self.trace = Trace()
        labels = jnp.array(labels, dtype=int)
        samples = jnp.array(samples, dtype=jnp.float32)
        batch_size = kwargs.get("batch_size", 1024)
        epochs = kwargs.get("epochs", 10)
        epochs *= batches_per_epoch

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
            if (k + 1) % 50 == 0:
                ma = jnp.mean(jnp.array(losses[-50:]))
                self.trace.losses.append(ma)
                tepochs.set_postfix(loss=ma)
                self.trace.iteration += 1
                lr_scale = otu.tree_get(self.state, "scale")
                self.trace.lr.append(lr_scale)

    def _init_state(self, **kwargs):
        """Initialise the training state and setup the optimizer."""
        dummy_x = jnp.zeros((1, self.ndims))
        _params = self.network.init(self.rng, dummy_x)
        lr = kwargs.get("lr", 1e-2)
        params = _params["params"]
        transition_steps = kwargs.get("transition_steps", 100)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                optax.cosine_decay_schedule(lr * 5, transition_steps, alpha=lr)
            ),
            optax.contrib.reduce_on_plateau(
                factor=0.5,
                patience=transition_steps // 10,  # 10
                # cooldown=transition_steps // 10,  # 10
                accumulation_size=transition_steps,
            ),
        )

        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=optimizer,
        )

    def fit(self, samples, labels, **kwargs):
        """Fit the classifier on provided samples.

        Args:
            samples (np.ndarray): Samples to train on.
            labels (np.array): integer class labels corresponding to each sample.

        Keyword Args:
            restart (bool): If True, reinitialise the model before training. Defaults to False.
            batch_size (int): Size of the training batches. Defaults to 1024.
            epochs (int): Number of training epochs. Defaults to 10.
            lr (float): Learning rate. Defaults to 1e-2.
            transition_steps (int): Number of steps to transition the learning rate.
                                    Defaults to 100.
        """
        restart = kwargs.get("restart", False)
        batch_size = kwargs.get("batch_size", 1024)
        data_size = samples.shape[0]
        batches_per_epoch = data_size // batch_size
        self.ndims = samples.shape[-1]
        if (not self.state) | restart:
            self._init_state(**kwargs)
        self._train(samples, labels, batches_per_epoch, **kwargs)
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


class Regressor(Classifier):
    """Regressor class wrapping a basic jax multiclass regressor."""

    def loss(self, params, batch, labels):
        """Loss function for training the calibrator."""
        output = self.state.apply_fn({"params": params}, batch)
        loss = optax.squared_error(output.squeeze(), labels).mean()
        return loss
