from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution
from scipy.stats import qmc


OPTIMIZER_REGISTRY: dict[str, type] = {}


def register_optimizer(name: str, optimizer_cls: type) -> None:
    OPTIMIZER_REGISTRY[name.lower()] = optimizer_cls


def create_optimizer(name: str, **kwargs):
    optimizer_cls = OPTIMIZER_REGISTRY.get(name.lower())
    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizer_cls(**kwargs)


@dataclass(slots=True)
class DifferentialEvolutionOptimizer:
    """Small adapter around scipy differential evolution."""

    strategy: str = "best1bin"
    popsize: int = 8
    tol: float = 1e-7
    mutation: tuple[float, float] = (0.5, 1.5)
    recombination: float = 0.7
    disp: bool = False
    seed: int = 42
    polish: bool = False
    atol: float = 1e-4
    updating: str = "deferred"
    workers: object | None = None
    map_func: Callable | None = None

    def optimize(
        self,
        objective: Callable,
        bounds,
        *,
        callback: Callable | None = None,
        maxiter: int = 40,
    ):
        return differential_evolution(
            objective,
            bounds=bounds,
            strategy=self.strategy,
            maxiter=maxiter,
            popsize=self.popsize,
            tol=self.tol,
            mutation=self.mutation,
            recombination=self.recombination,
            disp=self.disp,
            callback=callback,
            workers=self.map_func or self.workers or self._serial_map,
            seed=self.seed,
            polish=self.polish,
            atol=self.atol,
            updating=self.updating,
        )

    @staticmethod
    def _serial_map(func: Callable, iterable: Iterable):
        return list(map(func, iterable))


@dataclass(slots=True)
class SamplingOptimizer:
    """Sample the bounded parameter space directly for Monte Carlo style calibration."""

    sampler: str = "sobol"
    n_samples: int | None = None
    seed: int = 42
    callback_interval: int = 10

    def optimize(
        self,
        objective: Callable,
        bounds,
        *,
        callback: Callable | None = None,
        maxiter: int = 40,
    ) -> OptimizeResult:
        n_samples = int(self.n_samples if self.n_samples is not None else maxiter)
        if n_samples <= 0:
            raise ValueError("SamplingOptimizer requires a positive sample count.")

        lower = np.asarray([float(bound[0]) for bound in bounds], dtype=float)
        upper = np.asarray([float(bound[1]) for bound in bounds], dtype=float)
        unit_samples = self._sample_unit_hypercube(n_dim=len(bounds), n_samples=n_samples)
        samples = qmc.scale(unit_samples, lower, upper)

        best_x: np.ndarray | None = None
        best_fun = float("inf")
        best_index = -1
        stopped_early = False

        for idx, params in enumerate(samples):
            loss = float(objective(params))
            improved = best_x is None or loss < best_fun
            if improved:
                best_x = np.asarray(params, dtype=float).copy()
                best_fun = loss
                best_index = idx

            should_callback = (
                callback is not None
                and (
                    improved
                    or idx == 0
                    or (idx + 1) % max(1, self.callback_interval) == 0
                    or idx + 1 == n_samples
                )
            )
            if should_callback:
                should_stop = callback(
                    np.asarray(params, dtype=float),
                    None,
                    {
                        "sample_index": idx + 1,
                        "n_samples": n_samples,
                        "best_index": best_index + 1,
                        "best_loss": best_fun,
                    },
                )
                if should_stop:
                    stopped_early = True
                    break

        if best_x is None:
            raise RuntimeError("SamplingOptimizer did not evaluate any candidate.")

        success = np.isfinite(best_fun) and not stopped_early
        if stopped_early:
            message = "Sampling terminated early by callback."
        elif success:
            message = f"Completed {n_samples} {self.sampler} samples."
        else:
            message = f"Completed {n_samples} {self.sampler} samples but did not find a finite objective value."

        return OptimizeResult(
            x=best_x,
            fun=best_fun,
            success=success,
            message=message,
            nit=n_samples if not stopped_early else idx + 1,
            nfev=n_samples if not stopped_early else idx + 1,
            sampler=self.sampler,
            n_samples=n_samples,
            best_index=best_index,
        )

    def _sample_unit_hypercube(self, *, n_dim: int, n_samples: int) -> np.ndarray:
        sampler_name = self.sampler.lower()
        if sampler_name == "sobol":
            engine = qmc.Sobol(d=n_dim, scramble=True, seed=self.seed)
            power = int(np.ceil(np.log2(max(1, n_samples))))
            return engine.random_base2(m=power)[:n_samples]
        if sampler_name == "lhs":
            engine = qmc.LatinHypercube(d=n_dim, seed=self.seed)
            return engine.random(n=n_samples)
        if sampler_name == "random":
            rng = np.random.default_rng(self.seed)
            return rng.random((n_samples, n_dim))
        raise ValueError(f"Unknown sampling method: {self.sampler}")


register_optimizer("differential_evolution", DifferentialEvolutionOptimizer)
register_optimizer("glue", SamplingOptimizer)
register_optimizer("sampling", SamplingOptimizer)
