## Hilbert Curve Sampler for Optuna


[Optuna](https://optuna.org) is one of the most popular tools for hyperparameter tuning frameworks.

Optuna features a lot of tool, including a large vaiety of [samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html), including TPESampler, BoTorchSampler, CmaEsSampler, and so on...

For one very specific task I decided to implement a new sampler based on space-filling [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve).

![Hilber curve](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Hilbert-curve_rounded-gradient-animated.gif/440px-Hilbert-curve_rounded-gradient-animated.gif)



## optuna.samplers.HilbertSampler

Sampling over the Hilbert space filling curve.

This sampler is based on *independent sampling*.
See also `optuna.samplers.BaseSampler` for more details of 'independent sampling'.

The sampler was tested using Kurobako benchmark.

**Parameters:**
>**search_bounds** - A dictionary whose key and value are a parameter name and the corresponding search boundaries respectively.
**n_trials** - A number of points on the Hilber curve to be evaluated.
**seed** - A seed to fix the order of trials as the Hilbert curve points are randomly shuffled. 


**Example:**

```python

import optuna
from optuna.samplers import HilbertSampler

def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x**2

n_trial = 10
search_bounds = {"x": [-5, 5]}
study = optuna.create_study(sampler=HilbertSampler(search_bounds, n_trials))
study.optimize(objective, n_trials=10)
```
    

