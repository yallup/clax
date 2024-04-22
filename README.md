# clax

handy classifiers in jax

# Installation

`pip install .`

# Simple usage
```python
from clax import Classifier
from scipy.stats import multivariate_normal
import numpy as np

np.random.seed(2024)

dim = 5
n_sample = 100000

M_0 = multivariate_normal(mean=np.random.randn(dim))
M_1 = multivariate_normal(mean=np.random.randn(dim))

D_0 = M_0.rvs(n_sample)
D_1 = M_1.rvs(n_sample)

data = np.concatenate((D_0, D_1))
labels = np.concatenate((np.zeros(n_sample), np.ones(n_sample)))

D_test = M_1.rvs()

# Arg is the number classes
classifier = Classifier(2)
classifier.fit(data, labels, epochs=20)

# Default predict the logprobs of the data
pred = classifier.predict(D_test)

# use to predict the ratios
print(pred[0] - pred[1])
print(M_0.logpdf(D_test) - M_1.logpdf(D_test))

# or predict the probs
print(classifier.predict(D_test, log=False))
```

## Extension to regression
```python
from clax import Regressor
import matplotlib.pyplot as plt

regressor = Regressor(1)
D = M_0.rvs(n_sample)
target = M_0.logpdf(D)
regressor.fit(D, target, epochs=20)

D_test = M_0.rvs(1000)
target_test = M_0.logpdf(D_test).reshape(-1,1)

pred = regressor.predict(D_test).reshape(-1,1)
plt.plot(pred, target_test, "o")
plt.show()
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
