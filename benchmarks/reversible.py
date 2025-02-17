import os
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

jax.config.update("jax_enable_x64", True)


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, y_dim, width_size, depth, key):
        self.mlp = eqx.nn.MLP(y_dim, y_dim, width_size, depth, key=key)

    def __call__(self, t, y, args):
        return self.mlp(y)


@eqx.filter_jit
def solve(model, y0, adjoint):
    term = dfx.ODETerm(model)
    solver = dfx.Euler()
    t0 = 0.0
    t1 = 5.0
    dt0 = 0.01
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=adjoint,
        max_steps=500,
    )
    return sol.ys


@eqx.filter_value_and_grad
def grad_loss(model, y0, adjoint):
    ys = solve(model, y0, adjoint)
    return jnp.mean(ys**2)


def measure_runtime(y0, model, adjoint):
    tic = time.time()
    loss, grads = grad_loss(model, y0, adjoint)
    toc = time.time()
    print(f"Compile time: {(toc - tic):.5f}")

    repeats = 10
    tic = time.time()
    for i in range(repeats):
        loss, grads = jax.block_until_ready(grad_loss(model, y0, adjoint))
    toc = time.time()
    print(f"Runtime: {((toc - tic) / repeats):.5f}")


def ydim():
    y_dim = 10
    depth = 4
    model = VectorField(y_dim, y_dim, depth, key=jr.PRNGKey(10))

    print(f"y_dim = {y_dim}")
    print("--------------")
    y0 = jnp.linspace(1.0, 10.0, num=y_dim)
    print("Recursive")
    adjoint = dfx.RecursiveCheckpointAdjoint()
    measure_runtime(y0, model, adjoint)
    print("Reversible")
    adjoint = dfx.ReversibleAdjoint()
    measure_runtime(y0, model, adjoint)

    y_dim = 100
    model = VectorField(y_dim, y_dim, depth, key=jr.PRNGKey(10))
    print(f"\ny_dim = {y_dim}")
    print("-----------------")
    y0 = jnp.linspace(1.0, 10.0, num=y_dim)
    print("Recursive")
    adjoint = dfx.RecursiveCheckpointAdjoint()
    measure_runtime(y0, model, adjoint)
    print("Reversible")
    adjoint = dfx.ReversibleAdjoint()
    measure_runtime(y0, model, adjoint)


def profile(adjoint):
    y_dim = 1000
    depth = 4
    model = VectorField(y_dim, y_dim, depth, key=jr.PRNGKey(10))
    y0 = jnp.linspace(1.0, 10.0, num=y_dim)

    grad_loss(model, y0, adjoint)


if __name__ == "__main__":
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        adjoint = dfx.ReversibleAdjoint()
        profile(adjoint)

    # y_dim = 1000
    # depth = 4
    # model = VectorField(y_dim, y_dim, depth, key=jr.PRNGKey(10))
    # y0 = jnp.linspace(1.0, 10.0, num=y_dim)
    # adjoint = dfx.RecursiveCheckpointAdjoint()
    # measure_runtime(y0, model, adjoint)

    # adjoint = dfx.ReversibleAdjoint()
    # measure_runtime(y0, model, adjoint)
