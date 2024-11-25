from collections.abc import Callable
from typing import cast, Optional, TypeAlias, TypeVar

from equinox.internal import ω
from jaxtyping import PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._solution import RESULTS, update_result
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver, AbstractSolver, AbstractWrappedSolver
from .runge_kutta import AbstractRungeKutta


ω = cast(Callable, ω)
_BaseSolverState = TypeVar("_BaseSolverState")
_SolverState: TypeAlias = tuple[_BaseSolverState, Y]


def _add_maybe_none(x, y):
    if x is None:
        return None
    else:
        return x + y


class Reversible(
    AbstractAdaptiveSolver[_SolverState], AbstractWrappedSolver[_SolverState]
):
    """
    Reversible solver method.

    Allows any solver ([`diffrax.AbstractSolver`][]) to be made
    algebraically reversible.

    The convergence order of the reversible solver is inherited from the wrapped
    solver.

    Backpropagation through the reversible solver implies very low memory usage and
    exact gradient calculation (up to floating point errors). This is implemented in
    [`diffrax.ReversibleAdjoint`][] and passed to [`diffrax.diffeqsolve`][] as
    `adjoint=diffrax.ReversibleAdjoint()`.
    """

    solver: AbstractSolver
    l: RealScalarLike = 0.999

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    @property
    def root_finder(self):
        return self.solver.root_finder  # pyright: ignore

    @property
    def root_find_max_steps(self):
        return self.solver.root_find_max_steps  # pyright: ignore

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        if isinstance(self.solver, AbstractRungeKutta):
            object.__setattr__(self.solver.tableau, "fsal", False)
            object.__setattr__(self.solver.tableau, "ssal", False)
        original_solver_init = self.solver.init(terms, t0, t1, y0, args)
        return (original_solver_init, y0)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        original_solver_state, z0 = solver_state

        step_z0, z_error, dense_info, original_solver_state, result1 = self.solver.step(
            terms, t0, t1, z0, args, original_solver_state, made_jump
        )
        y1 = (self.l * (ω(y0) - ω(z0)) + ω(step_z0)).ω

        step_y1, y_error, _, _, result2 = self.solver.step(
            terms, t1, t0, y1, args, original_solver_state, made_jump
        )
        z1 = (ω(y1) + ω(z0) - ω(step_y1)).ω

        solver_state = (original_solver_state, z1)
        result = update_result(result1, result2)

        return y1, _add_maybe_none(z_error, y_error), dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)
