import abc
import functools as ft
import warnings
from collections.abc import Callable, Iterable
from typing import Any, cast, Optional, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import optimistix.internal as optxi
from equinox.internal import ω

from ._heuristics import is_sde, is_unsafe_sde
from ._saveat import save_y, SaveAt, SubSaveAt
from ._solution import RESULTS
from ._solver import (
    AbstractItoSolver,
    AbstractReversibleSolver,
    AbstractRungeKutta,
    AbstractSRK,
    AbstractStratonovichSolver,
)
from ._term import AbstractTerm, AdjointTerm


ω = cast(Callable, ω)


def _is_none(x):
    return x is None


def _is_subsaveat(x: Any) -> bool:
    return isinstance(x, SubSaveAt)


def _nondiff_solver_controller_state(
    adjoint, init_state, passed_solver_state, passed_controller_state
):
    if passed_solver_state:
        name = (
            f"When using `adjoint={adjoint.__class__.__name__}()`, then `solver_state`"
        )
        solver_fn = ft.partial(
            eqxi.nondifferentiable,
            name=name,
        )
    else:
        solver_fn = lax.stop_gradient
    if passed_controller_state:
        name = (
            f"When using `adjoint={adjoint.__class__.__name__}()`, then "
            "`controller_state`"
        )
        controller_fn = ft.partial(
            eqxi.nondifferentiable,
            name=name,
        )
    else:
        controller_fn = lax.stop_gradient
    init_state = eqx.tree_at(
        lambda s: s.solver_state,
        init_state,
        replace_fn=solver_fn,
        is_leaf=_is_none,
    )
    init_state = eqx.tree_at(
        lambda s: s.controller_state,
        init_state,
        replace_fn=controller_fn,
        is_leaf=_is_none,
    )
    return init_state


def _only_transpose_ys(final_state):
    from ._integrate import SaveState

    is_save_state = lambda x: isinstance(x, SaveState)

    def get_ys(_final_state):
        return [
            s.ys
            for s in jtu.tree_leaves(_final_state.save_state, is_leaf=is_save_state)
        ]

    ys = get_ys(final_state)

    named_nondiff_entries = (
        "y",
        "tprev",
        "tnext",
        "solver_state",
        "controller_state",
        "dense_ts",
        "dense_infos",
    )
    named_nondiff_values = tuple(
        eqxi.nondifferentiable_backward(getattr(final_state, k), name=k, symbolic=False)
        for k in named_nondiff_entries
    )

    final_state = eqxi.nondifferentiable_backward(final_state, symbolic=False)

    get_named_nondiff_entries = lambda s: tuple(
        getattr(s, k) for k in named_nondiff_entries
    )
    final_state = eqx.tree_at(
        get_named_nondiff_entries, final_state, named_nondiff_values, is_leaf=_is_none
    )

    final_state = eqx.tree_at(get_ys, final_state, ys)
    return final_state


class AbstractAdjoint(eqx.Module):
    """Abstract base class for all adjoint methods."""

    @abc.abstractmethod
    def loop(
        self,
        *,
        args,
        terms,
        solver,
        stepsize_controller,
        event,
        saveat,
        t0,
        t1,
        dt0,
        max_steps,
        throw,
        init_state,
        passed_solver_state,
        passed_controller_state,
        progress_meter,
    ) -> Any:
        """Runs the main solve loop. Subclasses can override this to provide custom
        backpropagation behaviour; see for example the implementation of
        [`diffrax.BacksolveAdjoint`][].
        """

    # Eurgh, delayed imports to handle circular dependencies.
    #
    # `integrate.py` defines the forward pass. `adjoint.py` defines the backward pass.
    # These pretty much necessarily depend on each other:
    #
    # - diffeqsolve needs to know about AbstractAdjoint, since it's one its arguments.
    # - BacksolveAdjoint needs to know about how to integrate a differential equation,
    #   since that's what it does.
    #
    # As such we get a circular dependency. We resolve it by lazily importing from
    # `integrate.py`. For convenience we make them available as properties here so all
    # adjoint methods can access these.
    @property
    def _loop(self):
        from ._integrate import loop

        return loop

    @property
    def _diffeqsolve(self):
        from ._integrate import diffeqsolve

        return diffeqsolve


_inner_loop = jax.named_call(eqxi.while_loop, name="inner-loop")
_outer_loop = jax.named_call(eqxi.while_loop, name="outer-loop")


def _uncallable(*args, **kwargs):
    assert False


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    """Enables support for backpropagating through [`diffrax.diffeqsolve`][] by
    differentiating the numerical solution directly. This is sometimes known as
    "discretise-then-optimise", or described as "backpropagation through the solver".

    Uses a binomial checkpointing scheme to keep memory usage low.

    For most problems this is the preferred technique for backpropagating through a
    differential equation, and as such it is the default for [`diffrax.diffeqsolve`][].

    !!! info

        Note that this cannot be forward-mode autodifferentiated. (E.g. using
        `jax.jvp`.) Try using [`diffrax.ForwardMode`][] if you need forward-mode
        autodifferentiation, or [`diffrax.DirectAdjoint`][] if you need both forward and
        reverse-mode autodifferentiation.

    ??? cite "References"

        Selecting which steps at which to save checkpoints (and when this is done, which
        old checkpoint to evict) is important for minimising the amount of recomputation
        performed.

        The implementation here performs "online checkpointing", as the number of steps
        is not known in advance. This was developed in:

        ```bibtex
        @article{stumm2010new,
            author = {Stumm, Philipp and Walther, Andrea},
            title = {New Algorithms for Optimal Online Checkpointing},
            journal = {SIAM Journal on Scientific Computing},
            volume = {32},
            number = {2},
            pages = {836--854},
            year = {2010},
            doi = {10.1137/080742439},
        }

        @article{wang2009minimal,
            author = {Wang, Qiqi and Moin, Parviz and Iaccarino, Gianluca},
            title = {Minimal Repetition Dynamic Checkpointing Algorithm for Unsteady
                     Adjoint Calculation},
            journal = {SIAM Journal on Scientific Computing},
            volume = {31},
            number = {4},
            pages = {2549--2567},
            year = {2009},
            doi = {10.1137/080727890},
        }
        ```

        For reference, the classical "offline checkpointing" (also known as "treeverse",
        "recursive binary checkpointing", "revolve" etc.) was developed in:

        ```bibtex
        @article{griewank1992achieving,
            author = {Griewank, Andreas},
            title = {Achieving logarithmic growth of temporal and spatial complexity in
                     reverse automatic differentiation},
            journal = {Optimization Methods and Software},
            volume = {1},
            number = {1},
            pages = {35--54},
            year  = {1992},
            publisher = {Taylor & Francis},
            doi = {10.1080/10556789208805505},
        }

        @article{griewank2000revolve,
            author = {Griewank, Andreas and Walther, Andrea},
            title = {Algorithm 799: Revolve: An Implementation of Checkpointing for the
                     Reverse or Adjoint Mode of Computational Differentiation},
            year = {2000},
            publisher = {Association for Computing Machinery},
            volume = {26},
            number = {1},
            doi = {10.1145/347837.347846},
            journal = {ACM Trans. Math. Softw.},
            pages = {19--45},
        }
        ```
    """

    checkpoints: Optional[int] = None

    def loop(
        self,
        *,
        terms,
        saveat,
        init_state,
        max_steps,
        throw,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw, passed_solver_state, passed_controller_state
        if is_unsafe_sde(terms):
            raise ValueError(
                "`adjoint=RecursiveCheckpointAdjoint()` does not support "
                "`UnsafeBrownianPath`. Consider using `adjoint=ForwardMode()` "
                "instead."
            )
        if self.checkpoints is None and max_steps is None:
            inner_while_loop = ft.partial(_inner_loop, kind="lax")
            outer_while_loop = ft.partial(_outer_loop, kind="lax")
            msg = (
                "Cannot reverse-mode autodifferentiate when using "
                "`diffeqsolve(..., max_steps=None, adjoint=RecursiveCheckpointAdjoint(checkpoints=None))`. "  # noqa: E501
                "This is because JAX needs to know how much memory to allocate for "
                "saving the forward pass. You should either put a bound on the maximum "
                "number of steps, or explicitly specify how many checkpoints to use."
            )
        else:
            inner_while_loop = ft.partial(_inner_loop, kind="checkpointed")
            outer_while_loop = ft.partial(
                _outer_loop, kind="checkpointed", checkpoints=self.checkpoints
            )
            msg = None
        final_state = self._loop(
            terms=terms,
            saveat=saveat,
            init_state=init_state,
            max_steps=max_steps,
            inner_while_loop=inner_while_loop,
            outer_while_loop=outer_while_loop,
            **kwargs,
        )
        if msg is not None:
            final_state = eqxi.nondifferentiable_backward(
                final_state, msg=msg, symbolic=True
            )
        return final_state


RecursiveCheckpointAdjoint.__init__.__doc__ = """
**Arguments:**

- `checkpoints`: the number of checkpoints to save. The amount of memory used by the
    differential equation solve will be roughly equal to the number of checkpoints
    multiplied by the size of `y0`. You can speed up backpropagation by allocating more
    checkpoints. (So it makes sense to set as many checkpoints as you have memory for.)
    This value can also be set to `None` (the default), in which case it will be set to
    `log(max_steps)`, for which a theoretical result is available guaranteeing that
    backpropagation will take `O(n log n)` time in the number of steps `n <= max_steps`.

You must pass either `diffeqsolve(..., max_steps=...)` or
`RecursiveCheckpointAdjoint(checkpoints=...)` to be able to backpropagate; otherwise
the computation will not be autodifferentiable.
"""


class DirectAdjoint(AbstractAdjoint):
    """A variant of [`diffrax.RecursiveCheckpointAdjoint`][] that is also able to
    support forward-mode autodifferentiation, whilst being less computationally
    efficient. (Under-the-hood it is using several nested layers of `jax.lax.scan`s and
    `jax.checkpoint`s, so that the cost of the solve increases with `max_steps`, even
    if you don't need that many steps to perform the solve in practice.)

    !!! warning

        This method is not recommended! In practice you should almost always use either
        [`diffrax.RecursiveCheckpointAdjoint`][] or [`diffrax.ForwardMode`][], depending
        on whether you need reverse or forward mode autodifferentiation. As this method
        is far less computationally efficient, then in practice it is only useful if you
        really **really** need to be able to support both kinds of autodifferentiation.
    """

    def loop(
        self,
        *,
        solver,
        max_steps,
        terms,
        throw,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw, passed_solver_state, passed_controller_state
        # TODO: remove the `is_unsafe_sde` guard.
        # We need JAX to release bloops, so that we can deprecate `kind="bounded"`.
        if is_unsafe_sde(terms):
            kind = "lax"
            msg = (
                "Cannot reverse-mode autodifferentiate when using `UnsafeBrownianPath`."
            )
        elif max_steps is None:
            kind = "lax"
            msg = (
                "Cannot reverse-mode autodifferentiate when using "
                "`diffeqsolve(..., max_steps=None, adjoint=DirectAdjoint())`. "
                "This is because JAX needs to know how much memory to allocate for "
                "saving the forward pass. You should either put a bound on the maximum "
                "number of steps, or switch to "
                "`adjoint=RecursiveCheckpointAdjoint(checkpoints=...)`, with an "
                "explicitly specified number of checkpoints."
            )
        else:
            kind = "bounded"
            msg = None
        # Support forward-mode autodiff.
        # TODO: remove this hack once we can JVP through custom_vjps.
        if (
            isinstance(solver, (AbstractRungeKutta, AbstractSRK))
            and solver.scan_kind is None
        ):
            solver = eqx.tree_at(
                lambda s: s.scan_kind, solver, "bounded", is_leaf=_is_none
            )
        inner_while_loop = ft.partial(_inner_loop, kind=kind)
        outer_while_loop = ft.partial(_outer_loop, kind=kind)
        final_state = self._loop(
            **kwargs,
            solver=solver,
            max_steps=max_steps,
            terms=terms,
            inner_while_loop=inner_while_loop,
            outer_while_loop=outer_while_loop,
        )
        if msg is not None:
            final_state = eqxi.nondifferentiable_backward(
                final_state, msg=msg, symbolic=True
            )
        return final_state


def _vf(ys, residual, inputs):
    state_no_y, _ = residual
    t = state_no_y.tprev

    def _unpack(_y):
        (_y1,) = _y
        return _y1

    y = jtu.tree_map(_unpack, ys)
    args, terms, _, _, solver, _, _ = inputs
    return solver.func(terms, t, y, args)


def _solve(inputs):
    args, terms, self, kwargs, solver, saveat, init_state = inputs
    final_state, aux_stats = self._loop(
        **kwargs,
        args=args,
        terms=terms,
        solver=solver,
        saveat=saveat,
        init_state=init_state,
        inner_while_loop=ft.partial(_inner_loop, kind="lax"),
        outer_while_loop=ft.partial(_outer_loop, kind="lax"),
    )
    # Note that we use .ys not .y here. The former is what is actually returned
    # by diffeqsolve, so it is the thing we want to attach the tangent to.
    #
    # Note that `final_state.save_state` has type PyTree[SaveState]. To access `.ys`
    # we are assuming that this PyTree has trivial structure. This is the case because
    # of the guard in `ImplicitAdjoint` that `saveat` be `SaveAt(t1=True)`.
    return final_state.save_state.ys, (
        eqx.tree_at(lambda s: s.save_state.ys, final_state, None),
        aux_stats,
    )


# Unwrap jaxtyping decorator during tests, so that these are global functions.
# This is needed to ensure `optx.implicit_jvp` is happy.
if _vf.__globals__["__name__"].startswith("jaxtyping"):
    _vf = _vf.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]
if _solve.__globals__["__name__"].startswith("jaxtyping"):
    _solve = _solve.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


def _frozenset(x: Union[object, Iterable[object]]) -> frozenset[object]:
    try:
        iter_x = iter(x)  # pyright: ignore
    except TypeError:
        return frozenset([x])
    else:
        return frozenset(iter_x)


class ImplicitAdjoint(AbstractAdjoint):
    r"""Backpropagate via the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem#Statement_of_the_theorem).

    This is used when solving towards a steady state, typically using
    [`diffrax.Event`][] where the condition function is obtained by calling
    [`diffrax.steady_state_event`][]. In this case, the output of the solver is $y(θ)$
    for which $f(t, y(θ), θ) = 0$. (Where $θ$ corresponds to all parameters found
    through `terms` and `args`, but not `y0`.) Then we can skip backpropagating through
    the solver and instead directly compute
    $\frac{\mathrm{d}y}{\mathrm{d}θ} = - (\frac{\mathrm{d}f}{\mathrm{d}y})^{-1}\frac{\mathrm{d}f}{\mathrm{d}θ}$
    via the implicit function theorem.

    Observe that this involves solving a linear system with matrix given by the Jacobian
    `df/dy`.
    """  # noqa: E501

    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    tags: frozenset[object] = eqx.field(
        default_factory=frozenset, converter=_frozenset, static=True
    )

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        throw,
        init_state,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw

        # `is` check because this may return a Tracer from SaveAt(ts=<array>)
        if eqx.tree_equal(saveat, SaveAt(t1=True)) is not True:
            raise ValueError(
                "Can only use `adjoint=ImplicitAdjoint()` with "
                "`saveat=SaveAt(t1=True)`."
            )
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )
        inputs = (args, terms, self, kwargs, solver, saveat, init_state)
        ys, residual = optxi.implicit_jvp(
            _solve, _vf, inputs, self.tags, self.linear_solver
        )

        final_state_no_ys, aux_stats = residual
        # Note that `final_state.save_state` has type PyTree[SaveState]. To access `.ys`
        # we are assuming that this PyTree has trivial structure. This is the case
        # because of the guard that `saveat` be `SaveAt(t1=True)`.
        final_state = eqx.tree_at(
            lambda s: s.save_state.ys, final_state_no_ys, ys, is_leaf=_is_none
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats


ImplicitAdjoint.__init__.__doc__ = """**Arguments:**

- `linear_solver`: A [Lineax](https://github.com/google/lineax) solver for solving the
    linear system.
- `tags`: Any Lineax [tags](https://docs.kidger.site/lineax/api/tags/) describing the
    Jacobian matrix `df/dy`.
"""


# Compute derivatives with respect to the first argument:
# - y, corresponding to the initial state;
# - args, corresponding to explicit parameters;
# - terms, corresponding to implicit parameters as part of the vector field.
@eqx.filter_custom_vjp
def _loop_backsolve(y__args__terms, *, self, throw, init_state, **kwargs):
    del throw
    y, args, terms = y__args__terms
    init_state = eqx.tree_at(lambda s: s.y, init_state, y)
    del y
    return self._loop(
        args=args,
        terms=terms,
        init_state=init_state,
        inner_while_loop=ft.partial(_inner_loop, kind="lax"),
        outer_while_loop=ft.partial(_outer_loop, kind="lax"),
        **kwargs,
    )


@_loop_backsolve.def_fwd
def _loop_backsolve_fwd(perturbed, y__args__terms, **kwargs):
    del perturbed
    final_state, aux_stats = _loop_backsolve(y__args__terms, **kwargs)
    # Note that `final_state.save_state` has type `PyTree[SaveState]`; here we are
    # relying on the guard in `BacksolveAdjoint` that it have trivial structure.
    ts = final_state.save_state.ts
    ys = final_state.save_state.ys
    return (final_state, aux_stats), (ts, ys)


def _materialise_none(y, grad_y):
    if grad_y is None and eqx.is_inexact_array(y):
        return jnp.zeros_like(y)
    else:
        return grad_y


@_loop_backsolve.def_bwd
def _loop_backsolve_bwd(
    residuals,
    grad_final_state__aux_stats,
    perturbed,
    y__args__terms,
    *,
    self,
    solver,
    stepsize_controller,
    event,
    saveat,
    t0,
    t1,
    dt0,
    max_steps,
    throw,
    init_state,
    progress_meter,
):
    assert event is None

    #
    # Unpack our various arguments. Delete a lot of things just to make sure we're not
    # using them later.
    #

    del perturbed, init_state, t1, progress_meter
    ts, ys = residuals
    del residuals
    grad_final_state, _ = grad_final_state__aux_stats
    # Note that `grad_final_state.save_state` has type `PyTree[SaveState]`; here we are
    # relying on the guard in `BacksolveAdjoint` that it have trivial structure.
    grad_ys = grad_final_state.save_state.ys
    # We take the simple way out and don't try to handle symbolic zeros.
    grad_ys = jtu.tree_map(_materialise_none, ys, grad_ys)
    del grad_final_state, grad_final_state__aux_stats
    y, args, terms = y__args__terms
    del y__args__terms
    diff_args = eqx.filter(args, eqx.is_inexact_array)
    diff_terms = eqx.filter(terms, eqx.is_inexact_array)
    zeros_like_y = jtu.tree_map(jnp.zeros_like, y)
    zeros_like_diff_args = jtu.tree_map(jnp.zeros_like, diff_args)
    zeros_like_diff_terms = jtu.tree_map(jnp.zeros_like, diff_terms)
    del diff_args, diff_terms
    # TODO: have this look inside MultiTerms? Need to think about the math. i.e.:
    # is_leaf=lambda x: isinstance(x, AbstractTerm) and not isinstance(x, MultiTerm)
    adjoint_terms = jtu.tree_map(
        AdjointTerm, terms, is_leaf=lambda x: isinstance(x, AbstractTerm)
    )
    diffeqsolve = self._diffeqsolve
    kwargs = dict(
        args=args,
        adjoint=self,
        solver=solver,
        stepsize_controller=stepsize_controller,
        terms=adjoint_terms,
        dt0=None if dt0 is None else -dt0,
        max_steps=max_steps,
        throw=throw,
    )
    kwargs.update(self.kwargs)
    del self, solver, stepsize_controller, adjoint_terms, dt0, max_steps, throw
    del y, args, terms
    # Note that `saveat.subs` has type `PyTree[SubSaveAt]`. Here we use the assumption
    # (checked in `BacksolveAdjoint`) that it has trivial pytree structure.
    saveat_t0 = saveat.subs.t0
    del saveat

    #
    # Now run a scan backwards in time, diffeqsolve'ing between each pair of adjacent
    # timestamps.
    #

    def _scan_fun(_state, _vals, first=False):
        _t1, _t0, _y0, _grad_y0 = _vals
        _a0, _solver_state, _controller_state = _state
        _a_y0, _a_diff_args0, _a_diff_term0 = _a0
        _a_y0 = (_a_y0**ω + _grad_y0**ω).ω
        _aug0 = (_y0, _a_y0, _a_diff_args0, _a_diff_term0)

        _sol = diffeqsolve(
            t0=_t0,
            t1=_t1,
            y0=_aug0,
            solver_state=_solver_state,
            controller_state=_controller_state,
            made_jump=not first,  # Adding _grad_y0, above, is a jump.
            saveat=SaveAt(t1=True, solver_state=True, controller_state=True),
            **kwargs,
        )

        def __get(__aug):
            assert __aug.shape[0] == 1
            return __aug[0]

        _aug1 = ω(_sol.ys).call(__get).ω
        _, _a_y1, _a_diff_args1, _a_diff_term1 = _aug1
        _a1 = (_a_y1, _a_diff_args1, _a_diff_term1)
        _solver_state = _sol.solver_state
        _controller_state = _sol.controller_state

        return (_a1, _solver_state, _controller_state), None

    state = ((zeros_like_y, zeros_like_diff_args, zeros_like_diff_terms), None, None)
    del zeros_like_y, zeros_like_diff_args, zeros_like_diff_terms

    # We always start backpropagating from `ts[-1]`.
    # We always finish backpropagating at `t0`.
    #
    # We may or may not have included `t0` in `ts`. (Depending on the value of
    # SaveaAt(t0=...) on the forward pass.)
    #
    # For some of these options, we run _scan_fun once outside the loop to get access
    # to solver_state etc. of the correct PyTree structure.
    if saveat_t0:
        if len(ts) > 2:
            val0 = (ts[-2], ts[-1], ω(ys)[-1].ω, ω(grad_ys)[-1].ω)
            state, _ = _scan_fun(state, val0, first=True)
            vals = (
                ts[:-2],
                ts[1:-1],
                ω(ys)[1:-1].ω,
                ω(grad_ys)[1:-1].ω,
            )
            state, _ = lax.scan(_scan_fun, state, vals, reverse=True)

        elif len(ts) == 1:
            # nothing to do, diffeqsolve is the identity when merely SaveAt(t0=True).
            pass

        else:
            assert len(ts) == 2
            val = (ts[0], ts[1], ω(ys)[1].ω, ω(grad_ys)[1].ω)
            state, _ = _scan_fun(state, val, first=True)

        aug1, _, _ = state
        a_y1, a_diff_args1, a_diff_terms1 = aug1
        a_y1 = (ω(a_y1) + ω(grad_ys)[0]).ω

    else:
        if len(ts) > 1:
            # TODO: fold this `_scan_fun` into the `lax.scan`. This will reduce compile
            # time.
            val0 = (ts[-2], ts[-1], ω(ys)[-1].ω, ω(grad_ys)[-1].ω)
            state, _ = _scan_fun(state, val0, first=True)
            vals = (
                jnp.concatenate([t0[None], ts[:-2]]),
                ts[:-1],
                ω(ys)[:-1].ω,
                ω(grad_ys)[:-1].ω,
            )
            state, _ = lax.scan(_scan_fun, state, vals, reverse=True)

        else:
            assert len(ts) == 1
            val = (t0, ts[0], ω(ys)[0].ω, ω(grad_ys)[0].ω)
            state, _ = _scan_fun(state, val, first=True)

        aug1, _, _ = state
        a_y1, a_diff_args1, a_diff_terms1 = aug1

    return a_y1, a_diff_args1, a_diff_terms1


class BacksolveAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by solving the continuous
    adjoint equations backwards-in-time. This is also sometimes known as
    "optimise-then-discretise", the "continuous adjoint method" or simply the "adjoint
    method".

    !!! warning

        This method is not recommended! It was popularised by
        [this paper](https://arxiv.org/abs/1806.07366), and for this reason it is
        sometimes erroneously believed to be a better method for backpropagation than
        other choices available.

        In practice whilst `BacksolveAdjoint` indeed has very low memory usage, its
        computed gradients will also be approximate. As the checkpointing of
        [`diffrax.RecursiveCheckpointAdjoint`][] also gives low memory usage, then in
        practice that is essentially always preferred.

    This will compute gradients with respect to the `terms`, `y0` and `args` arguments
    passed to [`diffrax.diffeqsolve`][]. If you attempt to compute gradients with
    respect to anything else (for example `t0`, or arguments passed via closure), then
    a `CustomVJPException` will be raised by JAX. See also
    [this FAQ](../../further_details/faq/#im-getting-a-customvjpexception)
    entry.

    !!! info

        Using this method prevents computing forward-mode autoderivatives of
        [`diffrax.diffeqsolve`][]. (That is to say, `jax.jvp` will not work.)
    """  # noqa: E501

    kwargs: dict[str, Any]

    def __init__(self, **kwargs):
        """
        **Arguments:**

        - `**kwargs`: The arguments for the [`diffrax.diffeqsolve`][] operations that
            are called on the backward pass. For example use
            ```python
            BacksolveAdjoint(solver=Dopri5())
            ```
            to specify a particular solver to use on the backward pass.
        """
        valid_keys = {
            "dt0",
            "solver",
            "stepsize_controller",
            "adjoint",
            "max_steps",
            "throw",
        }
        given_keys = set(kwargs.keys())
        diff_keys = given_keys - valid_keys
        if len(diff_keys) > 0:
            raise ValueError(
                "The following keyword argments are not valid for `BacksolveAdjoint`: "
                f"{diff_keys}"
            )
        self.kwargs = kwargs

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        init_state,
        passed_solver_state,
        passed_controller_state,
        event,
        **kwargs,
    ):
        if jtu.tree_structure(saveat.subs, is_leaf=_is_subsaveat) != jtu.tree_structure(
            0
        ):
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with `SaveAt(subs=...)`."
            )
        if saveat.dense or saveat.subs.steps:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with "
                "`saveat=SaveAt(steps=True)` or saveat=SaveAt(dense=True)`."
            )
        if saveat.subs.fn is not save_y:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with `saveat=SaveAt(fn=...)`."
            )
        if is_unsafe_sde(terms):
            raise ValueError(
                "`adjoint=BacksolveAdjoint()` does not support `UnsafeBrownianPath`. "
                "Consider using `adjoint=DirectAdjoint()` instead."
            )
        if is_sde(terms):
            if isinstance(solver, AbstractItoSolver):
                raise NotImplementedError(
                    f"`{solver.__class__.__name__}` converges to the Itô solution. "
                    "However `BacksolveAdjoint` currently only supports Stratonovich "
                    "SDEs."
                )
            elif not isinstance(solver, AbstractStratonovichSolver):
                warnings.warn(
                    f"{solver.__class__.__name__} is not marked as converging to "
                    "either the Itô or the Stratonovich solution. Note that "
                    "`BacksolveAdjoint` will only produce the correct solution for "
                    "Stratonovich SDEs."
                )
        if jtu.tree_structure(solver.term_structure) != jtu.tree_structure(0):
            raise NotImplementedError(
                "`diffrax.BacksolveAdjoint` is only compatible with solvers that take "
                "a single term."
            )
        if event is not None:
            raise NotImplementedError(
                "`diffrax.BacksolveAdjoint` is not compatible with events."
            )

        y = init_state.y
        init_state = eqx.tree_at(lambda s: s.y, init_state, object())
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )

        final_state, aux_stats = _loop_backsolve(
            (y, args, terms),
            self=self,
            saveat=saveat,
            init_state=init_state,
            solver=solver,
            event=event,
            **kwargs,
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats


class ForwardMode(AbstractAdjoint):
    """Enables support for forward-mode automatic differentiation (like `jax.jvp` or
    `jax.jacfwd`) through [`diffrax.diffeqsolve`][]. (As such this shouldn't really be
    called an 'adjoint' method -- which is a word that refers to any kind of
    reverse-mode autodifferentiation. Ah well.)

    This is useful when we have many more outputs than inputs to a function - for
    instance during parameter inference for ODE models with least-squares solvers such
    as
    [`optimistix.LevenbergMarquardt`](https://docs.kidger.site/optimistix/api/least_squares/#optimistix.LevenbergMarquardt),
    that operate on the residuals.
    """  # noqa: E501

    def loop(
        self,
        *,
        solver,
        throw,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw, passed_solver_state, passed_controller_state
        inner_while_loop = eqx.Partial(_inner_loop, kind="lax")
        outer_while_loop = eqx.Partial(_outer_loop, kind="lax")
        # Support forward-mode autodiff.
        # TODO: remove this hack once we can JVP through custom_vjps.
        if (
            isinstance(solver, (AbstractRungeKutta, AbstractSRK))
            and solver.scan_kind is None
        ):
            solver = eqx.tree_at(lambda s: s.scan_kind, solver, "lax", is_leaf=_is_none)
        final_state = self._loop(
            solver=solver,
            inner_while_loop=inner_while_loop,
            outer_while_loop=outer_while_loop,
            **kwargs,
        )
        return final_state


# Reversible Adjoint custom vjp computes gradients w.r.t.
# - y, corresponding to the initial state;
# - args, corresponding to explicit parameters;
# - terms, corresponding to implicit parameters as part of the vector field.


@eqx.filter_custom_vjp
def _loop_reversible(y__args__terms, *, self, throw, max_steps, init_state, **kwargs):
    del throw
    y, args, terms = y__args__terms
    init_state = eqx.tree_at(lambda s: s.y, init_state, y)
    del y
    return self._loop(
        args=args,
        terms=terms,
        max_steps=max_steps,
        init_state=init_state,
        inner_while_loop=ft.partial(_inner_loop, kind="lax"),
        outer_while_loop=ft.partial(_outer_loop, kind="lax"),
        **kwargs,
    )


@_loop_reversible.def_fwd
def _loop_reversible_fwd(perturbed, y__args__terms, **kwargs):
    del perturbed
    final_state, aux_stats = _loop_reversible(y__args__terms, **kwargs)
    init_ts = final_state.reversible_init_ts
    ts = final_state.reversible_ts
    ts_final_index = final_state.reversible_save_index
    y1 = final_state.y
    save_state = final_state.save_state
    solver_state = final_state.solver_state
    return (final_state, aux_stats), (
        init_ts,
        ts,
        ts_final_index,
        y1,
        save_state,
        solver_state,
    )


@_loop_reversible.def_bwd
def _loop_reversible_bwd(
    residuals,
    grad_final_state__aux_stats,
    perturbed,
    y__args__terms,
    *,
    self,
    saveat,
    init_state,
    solver,
    event,
    **kwargs,
):
    assert event is None

    del perturbed, self, init_state, kwargs
    init_ts, ts, ts_final_index, y1, save_state, solver_state = residuals
    del residuals

    grad_final_state, _ = grad_final_state__aux_stats
    saveat_ts = save_state.ts
    ys = save_state.ys
    saveat_ts_index = save_state.saveat_ts_index - 1
    grad_ys = grad_final_state.save_state.ys
    grad_ys = jtu.tree_map(_materialise_none, ys, grad_ys)

    if saveat.subs.t1:
        grad_y1 = (ω(grad_ys)[-1]).ω
    else:
        grad_y1 = jtu.tree_map(jnp.zeros_like, y1)

    if saveat.subs.t0:
        saveat_ts_index = saveat_ts_index + 1

    del grad_final_state, grad_final_state__aux_stats

    y, args, terms = y__args__terms
    del y__args__terms

    diff_state = eqx.filter(solver_state, eqx.is_inexact_array)
    diff_args = eqx.filter(args, eqx.is_inexact_array)
    diff_terms = eqx.filter(terms, eqx.is_inexact_array)
    grad_state = jtu.tree_map(jnp.zeros_like, diff_state)
    grad_args = jtu.tree_map(jnp.zeros_like, diff_args)
    grad_terms = jtu.tree_map(jnp.zeros_like, diff_terms)
    del diff_args, diff_terms

    def grad_step(state):
        def forward_step(y0, solver_state, args, terms):
            y1, _, dense_info, new_solver_state, result = solver.step(
                terms, t0, t1, y0, args, solver_state, False
            )
            assert result == RESULTS.successful
            return y1, dense_info, new_solver_state

        (
            saveat_ts_index,
            ts_index,
            y1,
            solver_state,
            grad_y1,
            grad_state,
            grad_args,
            grad_terms,
        ) = state

        t1 = ts[ts_index]
        t0 = ts[ts_index - 1]

        y0, dense_info, solver_state, result = solver.backward_step(
            terms, t0, t1, y1, args, solver_state, False
        )
        assert result == RESULTS.successful

        # Pull gradients back through interpolation

        def interpolate(t, t0, t1, dense_info):
            interpolator = solver.interpolation_cls(t0=t0, t1=t1, **dense_info)
            return interpolator.evaluate(t)

        def _cond_fun(inner_state):
            saveat_ts_index, _ = inner_state
            return (saveat_ts[saveat_ts_index] >= t0) & (saveat_ts_index >= 0)

        def _body_fun(inner_state):
            saveat_ts_index, grad_dense_info = inner_state
            t = saveat_ts[saveat_ts_index]
            grad_y = (ω(grad_ys)[saveat_ts_index]).ω
            _, interp_vjp = eqx.filter_vjp(interpolate, t, t0, t1, dense_info)
            _, _, _, dgrad_dense_info = interp_vjp(grad_y)
            grad_dense_info = eqx.apply_updates(grad_dense_info, dgrad_dense_info)
            saveat_ts_index = saveat_ts_index - 1
            return saveat_ts_index, grad_dense_info

        grad_dense_info = jtu.tree_map(jnp.zeros_like, dense_info)
        inner_state = (saveat_ts_index, grad_dense_info)
        inner_state = eqxi.while_loop(_cond_fun, _body_fun, inner_state, kind="lax")
        saveat_ts_index, grad_dense_info = inner_state

        # Pull gradients back through forward step

        _, vjp_fn = eqx.filter_vjp(forward_step, y0, solver_state, args, terms)
        grad_y0, grad_state, dgrad_args, dgrad_terms = vjp_fn(
            (grad_y1, grad_dense_info, grad_state)
        )

        grad_args = eqx.apply_updates(grad_args, dgrad_args)
        grad_terms = eqx.apply_updates(grad_terms, dgrad_terms)

        ts_index = ts_index - 1

        return (
            saveat_ts_index,
            ts_index,
            y0,
            solver_state,
            grad_y0,
            grad_state,
            grad_args,
            grad_terms,
        )

    def cond_fun(state):
        ts_index = state[1]
        return ts_index > 0

    state = (
        saveat_ts_index,
        ts_final_index,
        y1,
        solver_state,
        grad_y1,
        grad_state,
        grad_args,
        grad_terms,
    )

    state = jax.lax.while_loop(cond_fun, grad_step, state)
    _, _, y0, _, grad_y0, grad_state, grad_args, grad_terms = state

    # Pull solver_state gradients back onto y0, args, terms.

    init_t0, init_t1 = init_ts
    _, init_vjp = eqx.filter_vjp(solver.init, terms, init_t0, init_t1, y0, args)
    dgrad_terms, _, _, dgrad_y0, dgrad_args = init_vjp(grad_state)
    grad_y0 = eqx.apply_updates(grad_y0, dgrad_y0)
    grad_terms = eqx.apply_updates(grad_terms, dgrad_terms)
    grad_args = eqx.apply_updates(grad_args, dgrad_args)

    return grad_y0, grad_args, grad_terms


class ReversibleAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] when using a reversible solver
    [`diffrax.AbstractReversibleSolver`][].

    Gradient calculation is exact (up to floating point errors) and backpropagation
    becomes linear in time $O(n)$ and constant in memory $O(1)$, for $n$ time steps.

    !!! note

        This adjoint can be less numerically stable than
        [`diffrax.RecursiveCheckpointAdjoint`][] and [`diffrax.DirectAdjoint`][].
        Stability can be largely improved by using [double (64bit) precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision)
        and [smaller/adaptive step sizes](https://docs.kidger.site/diffrax/api/stepsize_controller/).

    ??? cite "References"

        For an introduction to reversible backpropagation, see these references:

        ```bibtex
        @article{mccallum2024efficient,
            title={Efficient, Accurate and Stable Gradients for Neural ODEs},
            author={McCallum, Sam and Foster, James},
            journal={arXiv preprint arXiv:2410.11648},
            year={2024}
        }

        @phdthesis{kidger2021on,
            title={{O}n {N}eural {D}ifferential {E}quations},
            author={Patrick Kidger},
            year={2021},
            school={University of Oxford},
        }
        ```
    """

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        max_steps,
        init_state,
        passed_solver_state,
        passed_controller_state,
        event,
        **kwargs,
    ):
        if not isinstance(solver, AbstractReversibleSolver):
            raise ValueError(
                "`ReversibleAdjoint` can only be used with an "
                "`AbstractReversibleSolver`"
            )
        if max_steps is None:
            raise ValueError(
                "`max_steps=None` is incompatible with `ReversibleAdjoint`."
            )

        if (
            jtu.tree_structure(saveat.subs, is_leaf=_is_subsaveat)
            != jtu.tree_structure(0)
            or saveat.dense
            or saveat.subs.steps
            or (saveat.subs.fn is not save_y)
        ):
            raise ValueError(
                "`ReversibleAdjoint` is only compatible with the following `SaveAt` "
                "properties: `t0`, `t1`, `ts`, `fn=save_y` (default)."
            )

        if event is not None:
            raise NotImplementedError(
                "`ReversibleAdjoint` is not compatible with events."
            )

        if is_unsafe_sde(terms):
            raise ValueError(
                "`ReversibleAdjoint` does not support `UnsafeBrownianPath`. "
                "Consider using `VirtualBrownianTree` instead."
            )

        y = init_state.y
        init_state = eqx.tree_at(lambda s: s.y, init_state, object())
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )

        final_state, aux_stats = _loop_reversible(
            (y, args, terms),
            self=self,
            saveat=saveat,
            max_steps=max_steps,
            init_state=init_state,
            solver=solver,
            event=event,
            **kwargs,
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats
