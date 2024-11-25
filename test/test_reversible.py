from typing import cast

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
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


# The adjoint tests look wrong at first glance so here's an explanation:
# We want to check that the gradients calculated by ReversibleAdjoint
# are the same as those calculated by RecursiveCheckpointAdjoint.
#
# The test looks weird because ReversibleAdjoint auto-wraps the solver
# to create a reversible version. So we use base_solver + ReversibleAdjoint and
# reversible_solver + RecursiveCheckpointAdjoint.
def test_reversible_adjoint():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    base_solver = diffrax.Tsit5()
    reversible_solver = diffrax.Reversible(base_solver)

    # Save y1 only
    saveat = diffrax.SaveAt(t1=True)

    # Constant steps
    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)

    # Adaptive steps
    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)

    # Save steps
    saveat = diffrax.SaveAt(steps=True)

    # Constant steps
    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)

    # Adaptive steps
    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)

    # Save steps (including t0)
    saveat = diffrax.SaveAt(t0=True, steps=True)

    # Constant steps
    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)

    # Adaptive steps
    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)


def test_implicit_solvers():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    base_solver = diffrax.Kvaerno5()
    reversible_solver = diffrax.Reversible(base_solver)

    # Save y1 only
    saveat = diffrax.SaveAt(t1=True)

    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)


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
    reversible_solver = diffrax.Reversible(base_solver)

    # Save y1 only
    saveat = diffrax.SaveAt(t1=True)

    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)


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
    reversible_solver = diffrax.Reversible(base_solver)

    # Save y1 only
    saveat = diffrax.SaveAt(t1=True)

    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)
