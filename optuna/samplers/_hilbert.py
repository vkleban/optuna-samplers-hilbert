import numpy as np

from typing import Any
from typing import Dict
from typing import Optional
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
import numpy.typing as npt

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

import warnings
from optuna.logging import get_logger

from hilbertcurve.hilbertcurve import HilbertCurve


_logger = get_logger(__name__)


class HilbertSampler(BaseSampler):
    """Sampling over the Hilbert space filling curve.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import HilbertSampler


            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                return x**2


            n_trial = 10
            search_bounds = {"x": [-5, 5]}
            study = optuna.create_study(sampler=HilbertSampler(search_bounds, n_trials))
            study.optimize(objective, n_trials=10)

    Args:
        search_bounds:
            A dictionary whose key and value are a parameter name and the corresponding search boundaries
            respectively.
        n_trials:
            A number of points on the Hilber curve to be evaluated.
        seed:
            A seed to fix the order of trials as the Hilbert curve points are randomly shuffled. 
    """

    @staticmethod
    def _softmax(x:Sequence[float], T: Optional[float]=1.0) -> npt.NDArray[np.float32]:
    #def _softmax(x, T):
        x = x - np.max(x)
        return np.exp(x/T) / np.exp(x/T).sum()  

    @staticmethod
    def _build_hilbert_curve(bounds, n_iter: int) -> npt.NDArray[np.float32]:
        bounds = np.array(bounds)

        n = bounds.shape[0]
        p = np.ceil(np.log2(n_iter + 1) / np.log2(2) / n)
        curve = HilbertCurve(p, n)
        
        max, min = np.max(bounds, axis=1), np.min(bounds, axis=1)

        pts = np.array(curve.points_from_distances(np.arange(n_iter)))
        scaler = pts.max(axis=0) - pts.min(axis=0)
        scaler[scaler == 0] = 1
        pts = (pts - pts.min(axis=0)) / scaler

        return np.squeeze(pts * (max - min) + min)

    def __init__(
        self, search_bounds: Mapping[str, Sequence[float]], n_trials: int, seed: Optional[int] = None
    ) -> None:
        self._search_space = {}

        self._param_names = sorted(search_bounds.keys())
        bounds = [search_bounds[k] for k in self._param_names]

        self._curve = self._build_hilbert_curve(bounds, n_trials)
        self._max_distance = n_trials
        self._rng = np.random.default_rng(seed)

    def reseed_rng(self) -> None:
        self._rng.rng.seed()

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # When the trial is created by RetryFailedTrialCallback or enqueue_trial, we should not
        # assign a new hilbert_distance.
        if ("hilbert_distance" in trial.system_attrs or "fixed_params" in trial.system_attrs):
            return

        if 0 <= trial.number and trial.number < self._max_distance:
            self._search_space = {}
            distance = trial.number
            for idx, param in enumerate(self._param_names):
                self._search_space[param] = self._curve[distance][idx]
            study._storage.set_trial_system_attr(
                trial._trial_id, "search_space", self._search_space
            )
            study._storage.set_trial_system_attr(
                trial._trial_id, "hilbert_distance", trial.number
            )

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        pass

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        if "hilbert_distance" not in trial.system_attrs:
            message = "All parameters must be specified when using HilbertSampler with enqueue_trial."
            raise ValueError(message)

        if param_name not in self._search_space:
            message = f"The parameter name, `{param_name}`, is not found in the given hilbert curve."
            raise ValueError(message)

        distance = trial.system_attrs["hilbert_distance"]
        param_value = self._curve[distance][self._param_names.index(param_name)]
        contains = param_distribution._contains(
            param_distribution.to_internal_repr(param_value)
        )
        if not contains:
            warnings.warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`."
            )

        return param_value
