import jax.numpy as jnp
import pytest
from jax import nn, random

from clax import Classifier, ClassifierSamples

rng = random.PRNGKey(0)

# @pytest.mark.parametrize("n_classes", [2, 10])
# class TestClassifier:
#     @pytest.fixture
#     def classifier(self, n_classes):
#         return Classifier(n_classes)

#     def test_fit(self, classifier, n_classes):
#         data_x = np.random.rand(100, 10)
#         data_y = np.random.randint(0, n_classes, 100)
#         classifier.fit(data_x, data_y)


@pytest.mark.parametrize("n_classes", [1, 2, 10])
def test_classifier(n_classes):
    classifier = Classifier(n_classes)
    data_x = random.uniform(rng, (100, 10))
    data_y = random.randint(rng, (100,), 0, n_classes)
    classifier.fit(data_x, data_y)
    y = classifier.predict(data_x)
    assert y.shape == (100, n_classes)
    assert jnp.isclose(nn.softmax(y).sum(axis=-1), 1).all()


def test_conditional_classifier():
    classifier = ClassifierSamples()
    data_x = random.normal(rng, (100, 10))
    data_y = random.normal(rng, (100, 10))
    classifier.fit(data_x, data_y)
    y = classifier.predict(data_x)
    assert y.shape == (100, 1)
