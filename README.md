# clax

handy classifiers in jax

# Installation

`pip install .`

# Simple usage
```python
from clax import Classifier
from scipy.stats import multivariate_normal
import numpy as np

dim = 5
n_sample = 100000

M_0 = multivariate_normal(mean=np.random.randn(dim))
M_1 = multivariate_normal(mean=np.random.randn(dim))

D_0 = M_0.rvs(n_sample)
D_1 = M_1.rvs(n_sample)

data = np.concatenate((D_0, D_1))
labels = np.concatenate((np.zeros(n_sample), np.ones(n_sample)))

D_test = M_1.rvs()

classifier = Classifier(2)
classifier.fit(data, labels, epochs=20)
```

# More advanced choices

```python
...

from flax import linen as nn
from clax.network import Network
import matplotlib.pyplot as plt


classifier = Classifier(2)

# Alter the default network:
network = Network(n_initial=512, n_hidden=32, n_layers=1, n_out=6)

classifier.network = network


# Or alternatively any flax network you like:
class CustomNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(6)(x)
        return x


# nb: we don't have 6 classes, the output dim should match the length of np.arange(num_classes)

classifier.network = CustomNetwork()
classifier.fit(data, labels, epochs=20)

plt.plot(classifier.trace.losses)
plt.show()


D_test = M_1.rvs(1000)
pred = classifier.predict(D_test)
network_pred = pred[..., 0] - pred[..., 1]

anlaytic_pred = M_0.logpdf(D_test) - M_1.logpdf(D_test)

plt.plot(network_pred, anlaytic_pred, "o")
plt.show()
```
