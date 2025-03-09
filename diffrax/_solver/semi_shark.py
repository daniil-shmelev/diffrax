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
        A_int, *rest = args
        drift, diffusion = terms.terms
        rv_dict = diffusion.contr(t0, t1, use_levy=True)

        dt = rv_dict.dt
        direction = jnp.where(dt > 0, 1, -1)
        W = direction * rv_dict.W
        H = direction * rv_dict.H
        alpha = 5 / 6
        dA_int = direction * (A_int(t1) - A_int(t0))
        A = dA_int / dt

        c1 = 1 / (1 - jnp.exp(-2 * dA_int)) - 1 / (2 * dA_int)
        c2 = (jnp.exp(dA_int) * (dA_int - 1) + 1) / (A * (jnp.exp(alpha * dA_int) - 1))

        # Shift rv's
        W_shift = (1 + 0.5 * dA_int) * W + dA_int * H
        H_shift = H - 1 / 12 * dA_int * W - 0.5 * dA_int * H

        # Diffusion
        g0 = diffusion.vf(t0, y0, args)
        g1 = diffusion.vf(t1, y0, args)
        dg = g1 - g0

        y0_shift = y0 + g0 * H_shift
        y1_shift = (
            jnp.exp(alpha * dA_int) * y0
            + (jnp.exp(alpha * dA_int) - 1) / A * drift.vf(t0, y0_shift, args)
            + c1 / c2 * dt * g0 * W_shift
            + jnp.exp(alpha * dA_int) * g0 * H_shift
        )

        y1 = (
            jnp.exp(dA_int) * y0
            + (jnp.exp(dA_int) - 1) / A * drift.vf(t0, y0_shift, args)
            + c2
            * (drift.vf(t0 + alpha * dt, y1_shift, args) - drift.vf(t0, y0_shift, args))
            + g0 * W_shift
            + dg * (0.5 * W - H)
        )
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: RealScalarLike, y0: Y, args: Args) -> VF:
        return terms.vf(t0, y0, args)
