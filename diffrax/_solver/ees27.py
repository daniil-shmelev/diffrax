from collections.abc import Callable
from typing import cast, ClassVar, Optional
from math import sqrt

import equinox as eqx
import equinox.internal as eqxi
from equinox.internal import ω
import numpy as np
from jaxtyping import PyTree

from .._local_interpolation import LocalLinearInterpolation, ThirdOrderHermitePolynomialInterpolation
from .base import AbstractStratonovichSolver, AbstractReversibleSolver
from .runge_kutta import AbstractERK, ButcherTableau

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._solution import RESULTS, update_result
from .._solver.base import (
    AbstractReversibleSolver,
    AbstractWrappedSolver,
)
from .._term import AbstractTerm

ω = cast(Callable, ω)
_SolverState = Y

SQRT2 = sqrt(2)

_ees27_tableau = ButcherTableau(
    a_lower=(
        np.array([(2 - SQRT2)/3]),
        np.array([(-4+SQRT2)/24, (4+SQRT2)/8]),
        np.array([(-176 + 145*SQRT2)/168, 3*(8-5*SQRT2)/56, 3*(3-SQRT2)/7]),
    ),
    b_sol=np.array([(5-3*SQRT2)/14, (3+SQRT2)/14, 3*(-1+2*SQRT2)/14, (9-4*SQRT2)/14]),
    b_error=np.array([1. - (5-3*SQRT2)/14, -(3+SQRT2)/14, -3*(-1+2*SQRT2)/14, -(9-4*SQRT2)/14]),
    c=np.array([(2-SQRT2)/3, (2+SQRT2)/6, (4+SQRT2)/6]),
)


class EES27(AbstractERK, AbstractReversibleSolver):
    """Explicit and Effectively Symmetric (EES) Runge-Kutta scheme of order 2
    and antisymmetric order 7, with parameter :math:`x = (5-3\\sqrt{2})/14`.
    """

    tableau: ClassVar[ButcherTableau] = _ees27_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k
    # interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
    #     LocalLinearInterpolation
    # )

    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 7

    def backward_step(
            self,
            terms: PyTree[AbstractTerm],
            t0: RealScalarLike,
            t1: RealScalarLike,
            y1: Y,
            args: Args,
            solver_state: _SolverState,
            made_jump: BoolScalarLike,
    ) -> tuple[Y, DenseInfo, _SolverState, RESULTS]:
        y0, _, dense_info, solver_state, result = self.step(terms, t0, t1, y1, args, solver_state, made_jump)
        return y0, dense_info, solver_state, result


eqxi.doc_remove_args("scan_kind")(EES27.__init__)
EES27.__init__.__doc__ = """**Arguments:** None"""
