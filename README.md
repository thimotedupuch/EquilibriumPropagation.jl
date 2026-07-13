# EquilibriumPropagation.jl

EquilibriumPropagation.jl is a compact, solver- and AD-backend-independent Julia
package for training conservative energy-based systems with equilibrium propagation
(EP). It finds free and nudged equilibria, then constructs parameter gradients from
the contrast between phases without differentiating through the equilibrium solver.

The package currently provides:

- one-sided and symmetric EP protocols;
- warm-started fixed-step relaxation with per-phase solver settings;
- explicit state and parameter AD backends through ADTypes and
  DifferentiationInterface;
- structured convergence and gradient diagnostics;
- an optional Optimisers.jl training extension; and
- continuous Hopfield examples for a two-dimensional spiral and MNIST classification.

## Installation

From Julia's package prompt:

```julia
pkg> add EquilibriumPropagation
```

Until the package is registered, develop it from a local checkout:

```julia
pkg> dev /path/to/EquilibriumPropagation.jl
```

An AD implementation must also be loaded by the application. For example:

```julia
using ADTypes, ForwardDiff
using EquilibriumPropagation

state_ad = AutoForwardDiff()
```

## Minimal workflow

```julia
model = EPModel(
    energy=(s, ps, x, st) -> sum(abs2, s) / 2 - sum(s .* (ps * x)),
    cost=(s, y, ps, st) -> sum(abs2, s .- y) / 2,
    readout=(s, ps, st) -> s,
    initial_state=(ps, x, st) -> zero(ps * x),
)

problem = EPProblem(model, parameters, NamedTuple(), input, target)
algorithm = EPAlgorithm(
    SymmetricEP(0.05),
    Relaxation(dt=0.2, maxiters=100, abstol=1e-6);
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoForwardDiff(),
)

gradients, stats = ep_gradient(problem, algorithm)
```

See the [getting-started guide](docs/src/getting_started.md),
[EP concepts](docs/src/concepts.md), and
[API reference](docs/src/api.md) for the complete interface.

## Continuous Hopfield spiral example

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/continuous_hopfield_spiral.jl
```

The script accepts options such as `--epochs=30`, `--points=400`,
`--batch-size=20`, `--hidden=32`, and `--seed=7`.

## MNIST classification example

```bash
julia --project=examples examples/mnist_classification.jl
```

The default run uses 7×7 average-pooled images, 1,000 training examples, and 500 test
examples. See the [MNIST example guide](docs/src/mnist.md) for command-line options.

## Testing

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```
