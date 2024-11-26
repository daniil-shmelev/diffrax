from typing import cast

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
import pytest
from jaxtyping import Array

from .helpers import tree_allclose


class _VectorField(eqx.Module):
    nondiff_arg: int
    diff_arg: float

    def __call__(self, t, y, args):
        assert y.shape == (2,)
        diff_arg, nondiff_arg = args
        dya = diff_arg * y[0] + nondiff_arg * y[1]
        dyb = self.nondiff_arg * y[0] + self.diff_arg * y[1]
        return jnp.stack([dya, dyb])


def test_constant_stepsizes():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    solver = diffrax.Tsit5()

    # Base
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
    )
    y1_base = sol.ys

    # Reversible
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        adjoint=diffrax.ReversibleAdjoint(),
    )
    y1_rev = sol.ys

    assert tree_allclose(y1_base, y1_rev, atol=1e-5)


def test_adaptive_stepsizes():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    # Base
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        stepsize_controller=stepsize_controller,
    )
    y1_base = sol.ys

    # Reversible
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=stepsize_controller,
    )
    y1_rev = sol.ys

    assert tree_allclose(y1_base, y1_rev, atol=1e-5)


# The adjoint comparison looks wrong at first glance so here's an explanation:
# We want to check that the gradients calculated by ReversibleAdjoint
# are the same as those calculated by RecursiveCheckpointAdjoint, for a fixed
# solver.
#
# The test looks weird because ReversibleAdjoint auto-wraps the solver
# to create a reversible version. So when calculating gradients we use
# base_solver + ReversibleAdjoint and reversible_solver + RecursiveCheckpointAdjoint,
# to ensure that the (reversible) solver is fixed and used across both adjoints.


@eqx.filter_value_and_grad
def _loss(y0__args__term, solver, saveat, adjoint, stepsize_controller):
    y0, args, term = y0__args__term

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
    )
    if eqx.tree_equal(saveat, diffrax.SaveAt(t1=True)) is True:
        y1 = sol.ys
    else:
        if eqx.tree_equal(saveat, diffrax.SaveAt(t0=True, steps=True)) is True:
            final_index = sol.stats["num_accepted_steps"] + 1
        elif eqx.tree_equal(saveat, diffrax.SaveAt(steps=True)) is True:
            final_index = sol.stats["num_accepted_steps"]

        ys = cast(Array, sol.ys)
        y1 = ys[:final_index]  # type: ignore

    return jnp.sum(cast(Array, y1))


def _compare_loss(y0__args__term, base_solver, saveat, stepsize_controller):
    reversible_solver = diffrax.Reversible(base_solver)

    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=stepsize_controller,
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=stepsize_controller,
    )
    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)


def test_reversible_adjoint():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    base_solver = diffrax.Tsit5()
    constant_steps = diffrax.ConstantStepSize()
    adaptive_steps = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    # Save y1 only
    saveat = diffrax.SaveAt(t1=True)
    _compare_loss(y0__args__term, base_solver, saveat, constant_steps)
    _compare_loss(y0__args__term, base_solver, saveat, adaptive_steps)

    # Save steps
    saveat = diffrax.SaveAt(steps=True)
    _compare_loss(y0__args__term, base_solver, saveat, constant_steps)
    _compare_loss(y0__args__term, base_solver, saveat, adaptive_steps)

    # Save steps (including t0)
    saveat = diffrax.SaveAt(t0=True, steps=True)
    _compare_loss(y0__args__term, base_solver, saveat, constant_steps)
    _compare_loss(y0__args__term, base_solver, saveat, adaptive_steps)


def test_implicit_solvers():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    base_solver = diffrax.Kvaerno5()
    adaptive_steps = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    saveat = diffrax.SaveAt(t1=True)
    _compare_loss(y0__args__term, base_solver, saveat, adaptive_steps)


def test_sde_additive_noise():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    drift = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    diffusion = lambda t, y, args: 1.0
    brownian_motion = diffrax.VirtualBrownianTree(
        0, 5, tol=1e-3, shape=(), levy_area=diffrax.SpaceTimeLevyArea, key=jr.PRNGKey(0)
    )
    terms = diffrax.MultiTerm(drift, diffrax.ControlTerm(diffusion, brownian_motion))
    y0__args__term = (y0, args, terms)
    del y0, args, terms

    base_solver = diffrax.ShARK()
    constant_steps = diffrax.ConstantStepSize()

    saveat = diffrax.SaveAt(t1=True)
    _compare_loss(y0__args__term, base_solver, saveat, constant_steps)


def test_sde_commutative_noise():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    drift = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    diffusion = lambda t, y, args: 0.5 * y
    brownian_motion = diffrax.VirtualBrownianTree(
        0, 5, tol=1e-3, shape=(), levy_area=diffrax.SpaceTimeLevyArea, key=jr.PRNGKey(0)
    )
    terms = diffrax.MultiTerm(drift, diffrax.ControlTerm(diffusion, brownian_motion))
    y0__args__term = (y0, args, terms)
    del y0, args, terms

    base_solver = diffrax.SlowRK()
    constant_steps = diffrax.ConstantStepSize()

    saveat = diffrax.SaveAt(t1=True)
    _compare_loss(y0__args__term, base_solver, saveat, constant_steps)


def test_events():
    def vector_field(t, y, args):
        _, v = y
        return jnp.array([v, -8.0])

    def cond_fn(t, y, args, **kwargs):
        x, _ = y
        return x

    @eqx.filter_value_and_grad
    def _event_loss(y0, adjoint):
        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0, adjoint=adjoint, event=event
        )
        return cast(Array, sol.ys)[0, 1]

    y0 = jnp.array([10.0, 0.0])
    t0 = 0
    t1 = jnp.inf
    dt0 = 0.1
    term = diffrax.ODETerm(vector_field)
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)
    solver = diffrax.Tsit5()

    msg = "`diffrax.ReversibleAdjoint` is not compatible with events."
    with pytest.raises(NotImplementedError, match=msg):
        v, grad_v = _event_loss(y0, adjoint=diffrax.ReversibleAdjoint())


def test_incorrect_saveat():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    base_solver = diffrax.Tsit5()

    # Save ts
    ts = jnp.linspace(0, 5)
    saveat_ts = diffrax.SaveAt(ts=ts)
    saveat_dense = diffrax.SaveAt(dense=True)
    saveat_t0 = diffrax.SaveAt(t0=True)
    saveat_fn = diffrax.SaveAt(ts=ts, fn=lambda t, y, args: t)

    with pytest.raises(ValueError):
        loss, grads_reversible = _loss(
            y0__args__term,
            base_solver,
            saveat_ts,
            adjoint=diffrax.ReversibleAdjoint(),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
    with pytest.raises(ValueError):
        loss, grads_reversible = _loss(
            y0__args__term,
            base_solver,
            saveat_dense,
            adjoint=diffrax.ReversibleAdjoint(),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
    with pytest.raises(ValueError):
        loss, grads_reversible = _loss(
            y0__args__term,
            base_solver,
            saveat_t0,
            adjoint=diffrax.ReversibleAdjoint(),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
    with pytest.raises(ValueError):
        loss, grads_reversible = _loss(
            y0__args__term,
            base_solver,
            saveat_fn,
            adjoint=diffrax.ReversibleAdjoint(),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
