Hilbert Curve Sampler for Optuna
================


[Optuna](https://optuna.org) is one of the most popular tools for hyperparameter tuning frameworks.

Optuna features a lot of tool, including a large vaiety of [samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html), including TPESampler, BoTorchSampler, CmaEsSampler, and so on...

For one very specific task I decided to implement a new sampler based on space-filling [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve).

![Hilber curve](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Hilbert-curve_rounded-gradient-animated.gif/440px-Hilbert-curve_rounded-gradient-animated.gif)

The sampler was tested using Kurobako benchmark.
