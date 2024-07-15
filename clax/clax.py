from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import jit
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
    """Classifier class wrapping a basic jax multiclass classifier.

    Takes labels and samples and trains a classifier to predict the labels.
    """

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
        self.n = n
        self.state = None
        self.dl = DataLoader
        if n == 1:
            self.loss_fn = optax.sigmoid_binary_cross_entropy
        else:
            self.loss_fn = optax.softmax_cross_entropy_with_integer_labels

    def loss(self, params, batch_stats, batch, labels, rng):
        """Loss function for training the classifier."""
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            batch,
            train=True,
            mutable=["batch_stats"],
        )
        loss = self.loss_fn(output.squeeze(), labels).mean()
        return loss, updates

    def _train(self, samples, labels, batches_per_epoch, **kwargs):
        """Internal wrapping of training loop."""
        self.trace = Trace()
        batch_size = kwargs.get("batch_size", 1024)
        epochs = kwargs.get("epochs", 10)
        # epochs *= batches_per_epoch

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

        dl = self.dl(samples.shape[0], labels.shape[0], **kwargs)
        tepochs = tqdm(range(epochs))
        for k in tepochs:
            epoch_losses = []
            for _ in range(batches_per_epoch):
                self.rng, step_rng = random.split(self.rng)
                perm, perm_label = dl.sample(batch_size)
                batch = samples[perm]
                batch_label = labels[perm_label]
                loss, self.state = update_step(self.state, batch, batch_label, step_rng)
                epoch_losses.append(loss)

            epoch_summary_loss = jnp.mean(jnp.asarray(epoch_losses))
            tepochs.set_postfix(loss="{:.2e}".format(epoch_summary_loss))
            losses.append(epoch_summary_loss)
            # if losses[::-1][:patience] < epoch_summary_loss:
            #     break
        self.trace.losses = jnp.asarray(losses)

    def _init_state(self, **kwargs):
        """Initialise the training state and setup the optimizer."""
        dummy_x = jnp.zeros((1, self.ndims))
        _params = self.network.init(self.rng, dummy_x, train=False)
        lr = kwargs.get("lr", 1e-2)
        optimizer = kwargs.get("optimizer", None)
        params = _params["params"]
        batch_stats = _params["batch_stats"]

        target_batches_per_epoch = kwargs.pop("target_batches_per_epoch")
        warmup_fraction = kwargs.get("warmup_fraction", 0.05)
        cold_fraction = kwargs.get("cold_fraction", 0.05)
        cold_lr = kwargs.get("cold_lr", 1e-3)
        epochs = kwargs.pop("epochs")

        self.schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=int(warmup_fraction * target_batches_per_epoch * epochs),
            decay_steps=int((1 - cold_fraction) * target_batches_per_epoch * epochs),
            end_value=lr * cold_lr,
            exponent=1.0,
        )
        if not optimizer:
            optimizer = optax.chain(
                optax.adaptive_grad_clip(0.01),
                # optax.contrib.schedule_free_adamw(lr, warmup_steps=transition_steps)
                optax.adamw(self.schedule),
                # optax.adamw(lr),
            )

        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            batch_stats=batch_stats,
            tx=optimizer,
        )

    def fit(self, samples, labels, epochs=10, **kwargs):
        """Fit the classifier on provided samples.

        Args:
            samples (np.ndarray): Samples to train on.
            labels (np.array): integer class labels corresponding to each sample.

        Keyword Args:
            restart (bool): If True, reinitialise the model before training. Defaults to False.
            batch_size (int): Size of the training batches. Defaults to 1024.
            epochs (int): Number of training epochs. Defaults to 10.
            lr (float): Learning rate. Defaults to 1e-2.

            optimizer (optax): Optimizer to use. Defaults to None. If none uses AdamW with a cosine schedule.
            with adjustable parameters as further kwargs:
                warmup_fraction (float): Fraction of the training steps to warm up the learning rate.
                                        Defaults to 0.05.
                cold_fraction (float): Fraction of the training steps at the cold learning rate.
                                        Defaults to 0.05.
                cold_lr (float): The factor to reduce learning rate to use during the cold phase. Defaults to 1e-3.
        """
        restart = kwargs.get("restart", False)
        batch_size = kwargs.get("batch_size", 1024)
        data_size = samples.shape[0]
        batches_per_epoch = data_size // batch_size
        self.ndims = samples.shape[-1]
        kwargs["epochs"] = epochs
        if (not self.state) | restart:
            kwargs["target_batches_per_epoch"] = batches_per_epoch
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
        """
        return self._predict_weight(samples)


class ClassifierSamples(Classifier):
    """Extension of basic Classifier to allow initialization of a binary classifier,
    with labels implicit from two piles of data.
    """

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
        loss = self.loss_fn(output.squeeze(), labels).mean()
        return loss, updates

    def fit(self, samples_a, samples_b, epochs=10, **kwargs):
        """Fit the classifier on provided samples.

        Args:
            samples (np.ndarray): Samples to train on.
            labels (np.array): integer class labels corresponding to each sample.


        Keyword Args:
            restart (bool): If True, reinitialise the model before training. Defaults to False.
            batch_size (int): Size of the training batches. Defaults to 1024.
            epochs (int): Number of training epochs. Defaults to 10.
            lr (float): Learning rate. Defaults to 1e-2.

            optimizer (optax): Optimizer to use. Defaults to None. If none uses AdamW with a cosine schedule.
            with adjustable parameters as further kwargs:
                warmup_fraction (float): Fraction of the training steps to warm up the learning rate.
                                        Defaults to 0.05.
                cold_fraction (float): Fraction of the training steps at the cold learning rate.
                                        Defaults to 0.05.
                cold_lr (float): The factor to reduce learning rate to use during the cold phase. Defaults to 1e-3.
        """
        restart = kwargs.get("restart", False)
        batch_size = kwargs.get("batch_size", 1024)
        self.ndims = kwargs.get("ndims", samples_a.shape[-1])
        data_size = samples_a.shape[0]
        batches_per_epoch = data_size // batch_size
        kwargs["epochs"] = epochs
        if (not self.state) | restart:
            kwargs["target_batches_per_epoch"] = batches_per_epoch
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


class Regressor(Classifier):
    """Regressor class wrapping a basic jax multiclass regressor."""

    def loss(self, params, batch, labels):
        """Loss function for training the regressor."""
        output = self.state.apply_fn({"params": params}, batch)
        loss = optax.squared_error(output.squeeze(), labels).mean()
        return loss
