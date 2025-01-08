import time

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class VectorField(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(1, 10, use_bias=True, key=key1),
            jnp.tanh,
            eqx.nn.Linear(10, 1, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers:
            y = layer(y)
        return y


@eqx.filter_value_and_grad
def grad_loss(y0__term, t1, adjoint, solver):
    y0, term = y0__term
    t0 = 0
    dt0 = 0.01
    max_steps = int((t1 - t0) / dt0)
    ys = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        adjoint=adjoint,
        max_steps=max_steps,
    ).ys

    return jnp.sum(ys**2)  # pyright: ignore


def run(adjoint, solver, t1):
    term = dfx.ODETerm(VectorField(jr.PRNGKey(0)))
    y0 = jnp.array([1.0])

    tic = time.time()
    grad_loss((y0, term), t1, adjoint, solver)
    toc = time.time()
    runtime = toc - tic

    return runtime


if __name__ == "__main__":
    adjoint = dfx.ReversibleAdjoint()
    solver = dfx.Midpoint()
    compiletime = run(adjoint, solver, t1=10)
    print(f"Compilation time: {compiletime:.3f}")

    runtime = run(adjoint, solver, t1=10)
    print(f"Runtime: {runtime:.3f}")
