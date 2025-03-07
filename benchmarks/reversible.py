import time
from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array


jax.config.update("jax_enable_x64", True)


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, key):
        self.mlp = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)

    def __call__(self, t, y, args):
        return args * self.mlp(y)


@eqx.filter_value_and_grad
def _loss(y0__args__term, solver, saveat, adjoint, stepsize_controller, dual_y0):
    y0, args, term = y0__args__term
    max_steps = len(stepsize_controller.ts)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=None,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    if dual_y0:
        y1 = sol.ys[0]  # pyright: ignore
    else:
        y1 = sol.ys
    return jnp.sum(cast(Array, y1))


def measure_runtime(y0__args__term, solver, adjoint, stepsize_controller, dual_y0):
    tic = time.time()
    loss, grads = _loss(
        y0__args__term,
        solver,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
        dual_y0=dual_y0,
    )
    toc = time.time()
    print(f"Compile time: {(toc - tic):.5f}")

    repeats = 10
    tic = time.time()
    for i in range(repeats):
        loss, grads = jax.block_until_ready(
            _loss(
                y0__args__term,
                solver,
                saveat=diffrax.SaveAt(t1=True),
                adjoint=adjoint,
                stepsize_controller=stepsize_controller,
                dual_y0=dual_y0,
            )
        )
    toc = time.time()
    print(f"Runtime: {((toc - tic) / repeats):.5f}")


if __name__ == "__main__":
    n = 10
    y0 = jnp.linspace(1, 10, num=n)
    key = jr.PRNGKey(10)
    f = VectorField(n, n, n, depth=2, key=key)
    terms = diffrax.ODETerm(f)
    args = jnp.linspace(0, 1, n)
    base_solver = diffrax.LeapfrogMidpoint()
    # solver = diffrax.Reversible(base_solver)
    solver = base_solver
    t0 = 0
    t1 = 5.0
    dt0 = 0.01
    ts = jnp.linspace(t0, t1, num=500)
    stepsize_controller = diffrax.StepTo(ts=ts)

    saveat = diffrax.SaveAt(ts=ts)
    # saveat = diffrax.SaveAt(t1=True)
    adjoint = diffrax.ReversibleAdjoint()
    loss, grads = _loss(
        (y0, args, terms), solver, saveat, adjoint, stepsize_controller, dual_y0=False
    )
    print(grads[0])
    # print(grads[2].vector_field.mlp.layers[0].weight)
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    loss, grads = _loss(
        (y0, args, terms),
        solver,
        saveat,
        adjoint,
        stepsize_controller,
        dual_y0=False,
    )
    print("---")
    print(grads[0])
    # print(grads[2].vector_field.mlp.layers[0].weight)

    # print("Recursive")
    # adjoint = diffrax.RecursiveCheckpointAdjoint()
    # measure_runtime(
    #     (y0, args, terms), base_solver, adjoint, stepsize_controller, dual_y0=False
    # )

    # print("Reversible")
    # adjoint = diffrax.ReversibleAdjoint()
    # measure_runtime(
    #     (y0, args, terms), solver, adjoint, stepsize_controller, dual_y0=False
    # )
