from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn
from flax.training import train_state
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

    def __init__(self, n=1, **kwargs):
        """Initialise the Classifier.

        args:
            n: int: Number of classes.
        kwargs:
            seed: int: Random seed.
        """
        self.rng = random.PRNGKey(kwargs.get("seed", 2024))
        self.n = n
        self.network = Network(n_out=n)
        self.state = None

    def loss(self, params, batch_stats, batch, labels, rng):
        """Loss function for training the classifier."""
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            batch,
            train=True,
            mutable=["batch_stats"],
        )
        # loss = optax.softmax_cross_entropy_with_integer_labels(
        #     output.squeeze(), labels
        # ).mean()
        loss = optax.sigmoid_binary_cross_entropy(output.squeeze(), labels).mean()
        return loss, updates

    def _train(self, samples, labels, batches_per_epoch, **kwargs):
        """Internal wrapping of training loop."""
        self.trace = Trace()
        batch_size = kwargs.get("batch_size", 1024)
        epochs = kwargs.get("epochs", 10)
        epochs *= batches_per_epoch

        @jit
        def update_step(state, samples, labels, rng):
            (val, updates), grads = jax.value_and_grad(self.loss, has_aux=True)(
                state.params, state.batch_stats, samples, labels, rng
            )
            state = state.apply_gradients(grads=grads)  # , scale_value=val)
            state = state.replace(batch_stats=updates["batch_stats"])
            return val, state

        train_size = samples.shape[0]
        batch_size = min(batch_size, train_size)
        losses = []
        map = DataLoader(samples, labels)
        tepochs = tqdm(range(epochs))
        for k in tepochs:
            self.rng, step_rng = random.split(self.rng)
            perm, _ = map.sample(batch_size)
            batch = samples[perm]
            batch_label = labels[perm]
            loss, self.state = update_step(self.state, batch, batch_label, step_rng)
            losses.append(loss)
            # self.state.losses.append(loss)
            if (k + 1) % 50 == 0:
                ma = jnp.mean(jnp.array(losses[-50:]))
                self.trace.losses.append(ma)
                tepochs.set_postfix(loss="{:.2e}".format(ma))
                self.trace.iteration += 1
                # lr_scale = otu.tree_get(self.state, "scale")
                # self.trace.lr.append(lr_scale)

    def _init_state(self, **kwargs):
        """Initialise the training state and setup the optimizer."""
        dummy_x = jnp.zeros((1, self.ndims))
        _params = self.network.init(self.rng, dummy_x, train=False)
        lr = kwargs.get("lr", 1e-2)
        params = _params["params"]
        batch_stats = _params["batch_stats"]
        transition_steps = kwargs.get("transition_steps", 1000)
        self.schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=transition_steps,
            decay_steps=transition_steps * 10,
            end_value=lr * 1e-4,
            exponent=1.0,
        )
        optimizer = optax.chain(
            # optax.clip_by_global_norm(1.0),
            optax.adaptive_grad_clip(0.01),
            # optax.adam(lr),
            # optax.adamw(self.schedule),
            optax.adamw(lr),
        )

        # self.state = train_state.TrainState.create(
        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            batch_stats=batch_stats,
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
        labels = jnp.array(labels, dtype=int)
        samples = jnp.array(samples, dtype=jnp.float32)
        self._train(samples, labels, batches_per_epoch, **kwargs)
        self._predict_weight = lambda x: self.state.apply_fn(
            {
                "params": self.state.params,
                "batch_stats": self.state.batch_stats,
            },
            x,
            train=False,
        )

    def predict(self, samples):
        """Predict the class (log) - probabilities for the provided samples.

        Args:
            samples (np.ndarray): Samples to predict on.
            log (bool): If True, return the log-probabilities. Defaults to True.
        """
        return self._predict_weight(samples)


class ConditionalClassifier(Classifier):
    def loss(self, params, batch_stats, batch, labels, rng):
        """Loss function for training the classifier."""

        batch = jnp.concatenate([batch, labels], axis=0)
        labels = jnp.concatenate(
            [jnp.ones(batch.shape[0] // 2), jnp.zeros(batch.shape[0] // 2)]
        )

        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            batch,
            train=True,
            mutable=["batch_stats"],
        )
        # loss = optax.softmax_cross_entropy_with_integer_labels(
        #     output.squeeze(), labels
        # ).mean()
        loss = optax.sigmoid_binary_cross_entropy(output.squeeze(), labels).mean()
        return loss, updates

    def fit(self, samples_a, samples_b, **kwargs):
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
        self.ndims = kwargs.get("ndims", samples_a.shape[-1])
        data_size = samples_a.shape[0]
        batches_per_epoch = data_size // batch_size
        if (not self.state) | restart:
            self._init_state(**kwargs)
        self._train(samples_a, samples_b, batches_per_epoch, **kwargs)
        self._predict_weight = lambda x: self.state.apply_fn(
            {
                "params": self.state.params,
                "batch_stats": self.state.batch_stats,
            },
            x,
            train=False,
        )

    def predict(self, samples):
        """Predict the class (log) - probabilities for the provided samples.

        Args:
            samples (np.ndarray): Samples to predict on.
            log (bool): If True, return the log-probabilities. Defaults to True.
        """
        return self._predict_weight(samples)


class Regressor(Classifier):
    """Regressor class wrapping a basic jax multiclass regressor."""

    def loss(self, params, batch, labels):
        """Loss function for training the regressor."""
        output = self.state.apply_fn({"params": params}, batch)
        loss = optax.squared_error(output.squeeze(), labels).mean()
        return loss
