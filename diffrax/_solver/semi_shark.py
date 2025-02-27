from collections.abc import Callable
from typing import Any, ClassVar
from typing_extensions import TypeAlias

import jax.numpy as jnp

from .._custom_types import (
    AbstractSpaceTimeLevyArea,
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, MultiTerm
from .base import AbstractStratonovichSolver


_SolverState: TypeAlias = None


class SemiShARK(AbstractStratonovichSolver):
    """Shifted Additive-noise Runge-Kutta method for Semi-linear SDEs.

    This assumes an SDE of the form dy = gamma*y + f(t, y)dt + dW_t.

    - gamma is passed as args=(gamma, ...)
    - f is passed as drift
    - diffusion coefficient is assumed to be 1.0
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation  # TODO use something better than this?

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5

    def init(
        self,
        terms: MultiTerm[
            tuple[
                AbstractTerm[Any, RealScalarLike],
                AbstractTerm[Any, AbstractSpaceTimeLevyArea],
            ]
        ],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: MultiTerm[
            tuple[
                AbstractTerm[Any, RealScalarLike],
                AbstractTerm[Any, AbstractSpaceTimeLevyArea],
            ]
        ],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        gamma, *rest = args
        drift, diffusion = terms.terms
        rv_dict = diffusion.contr(t0, t1, use_levy=True)
        dt = rv_dict.dt
        W = rv_dict.W
        H = rv_dict.H
        alpha = 5 / 6
        c1 = 1 / (1 - jnp.exp(-2 * gamma * dt)) - 1 / (2 * gamma * dt)
        c2 = (jnp.exp(gamma * dt) * (gamma * dt - 1) + 1) / (
            gamma * (jnp.exp(alpha * gamma * dt) - 1)
        )

        # Shift rv's
        W_shift = (1 + 0.5 * gamma * dt) * W + gamma * dt * H
        H_shift = H - 1 / 12 * gamma * dt * W - 0.5 * gamma * dt * H

        # Diffusion
        g0 = diffusion.vf(t0, y0, args)
        g1 = diffusion.vf(t1, y0, args)
        g_delta = g1 - g0

        y0_shift = y0 + g0 * H_shift
        y1_shift = (
            jnp.exp(alpha * gamma * dt) * y0
            + (jnp.exp(alpha * gamma * dt) - 1) / gamma * drift.vf(t0, y0_shift, args)
            + c1 / c2 * dt * g0 * W_shift
            + jnp.exp(alpha * gamma * dt) * g0 * H_shift
        )

        y1 = (
            jnp.exp(gamma * dt) * y0
            + (jnp.exp(gamma * dt) - 1) / gamma * drift.vf(t0, y0_shift, args)
            + c2
            * (drift.vf(t0 + alpha * dt, y1_shift, args) - drift.vf(t0, y0_shift, args))
            + g0 * W_shift
            + g_delta * (0.5 * W - H)
        )
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: RealScalarLike, y0: Y, args: Args) -> VF:
        return terms.vf(t0, y0, args)
