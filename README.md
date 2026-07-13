# EquilibriumPropagation.jl

EquilibriumPropagation.jl is a compact, solver- and AD-backend-independent Julia
package for training conservative energy-based and general dynamical systems with
equilibrium propagation (EP). It finds free and nudged equilibria, then constructs
parameter gradients without differentiating through the equilibrium solver.

The package currently provides:

- declarative continuous Hopfield network construction with named parameter trees;
- one-sided and symmetric EP protocols;
- warm-started fixed-step relaxation with per-phase solver settings;
- optional ODE, dynamic steady-state, and direct root relaxation wrappers;
- continual parameter updates coupled to nudged state relaxation;
- AsymEP and Dyadic EP for arbitrary differentiable non-conservative vector fields;
- Holomorphic EP with finite-radius complex nudging and Fourier gradient extraction;
- explicit state and parameter AD backends through ADTypes and
  DifferentiationInterface;
- structured convergence and gradient diagnostics;
- an optional Optimisers.jl training extension;
- optional Lux adapters for scalar energies and non-conservative vector fields; and
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

## Continuous Hopfield builder

```julia
using Random

network = ContinuousHopfield(
    (784, 128, 10);
    activation=tanh,
    potential=QuadraticPotential(),
    input_clamp=HardClamp(),
    output_cost=BipolarSquaredError(),
    bias=true,
    recurrent=false,
)

model, parameters = EquilibriumPropagation.setup(Random.default_rng(), network)
problem = EPProblem(model, parameters, NamedTuple(), images, targets)
```

The input remains hard-clamped outside the dynamical state. Parameters are returned
as `(weights=..., biases=..., recurrent=...)`, and the resulting `model` works with
the same solvers and EP protocols as a hand-written `EPModel`.

See the [getting-started guide](docs/src/getting_started.md),
[EP concepts](docs/src/concepts.md), and
[API reference](docs/src/api.md) for the complete interface.

The optional SciML integrations accept `Tsit5()`, `Rodas5P()`, and NonlinearSolve
algorithms explicitly. See the [solver guide](docs/src/solvers.md) for examples.

For asymmetric or feedforward dynamics, `DynamicalModel` keeps the vector-field
interface separate from conservative `EPModel`. See the
[non-conservative EP guide](docs/src/nonconservative_ep.md) for AsymEP and Dyadic EP.
The [Holomorphic EP guide](docs/src/holomorphic_ep.md) covers complex-valued model
requirements and finite Fourier sampling.
Lux users can keep explicit parameter and state trees through `lux_energy_model` and
`lux_dynamical_model`; see the [Lux integration guide](docs/src/lux.md).
The [feedforward MLP tutorial](docs/src/asymep_mlp.md) trains triangular Lux dynamics
with AsymEP, while the MNIST tutorial now uses a hidden Hopfield population.

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

## Feedforward AsymEP MLP tutorial

```bash
julia --project=examples examples/asymep_mlp.jl
```

This tutorial expresses a directed `2 → 16 → 2` Lux MLP as triangular equilibrium
dynamics and trains it on interleaved spirals with AsymEP. See the
[feedforward MLP guide](docs/src/asymep_mlp.md) for the equations and options.

## Testing

The repository has an explicit test environment so the optional extension tests can
run without creating a temporary Pkg test environment:

```bash
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
```
