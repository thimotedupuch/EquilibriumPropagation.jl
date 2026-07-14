# EquilibriumPropagation.jl

EquilibriumPropagation.jl is a composable Julia toolkit for training equilibrium
models with equilibrium propagation (EP). It covers conservative energy-based
networks, non-conservative dynamical systems, continual parameter updates, and
holomorphic finite-radius estimators. Models are ordinary Julia functions, while
solvers, differentiation backends, learning protocols, and optimizer updates remain
explicit and independently configurable.

The package supports small research models as well as compiled accelerator workflows.
Reactant can lower complete fixed-shape training steps—including equilibrium
relaxation, EnzymeMLIR differentiation, phase contrasts, convergence loops, and
Optimisers.jl state updates—to OpenXLA for CPU, GPU, and TPU execution.

## Core workflow

A conservative model supplies an internal energy ``E``, supervised cost ``C``,
readout, and initial state. For nudging strength ``\beta``, EP relaxes the total
energy

```math
F_\beta(s;\theta,x,y)
= E(s;\theta,x) + \beta C(s,y;\theta),
```

then constructs a parameter update from derivatives evaluated at free and nudged
equilibria. The equilibrium states are treated as constants during parameter
differentiation: EP does not backpropagate through the relaxation trajectory.

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff

algorithm = EPAlgorithm(
    SymmetricEP(0.05f0),
    Relaxation(dt=0.2f0, maxiters=100, abstol=1f-5);
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoForwardDiff(),
)

gradient, stats = ep_gradient(problem, algorithm)
```

[`EPStats`](@ref) reports loss, residuals, iteration counts, phase energies,
convergence, gradient norm, and phase displacement so numerical accuracy remains
visible to training code.

## Learning protocols

The conservative [`EPModel`](@ref) interface supports:

- [`OneSidedEP`](@ref), contrasting a positive nudged equilibrium with the free
  equilibrium;
- [`SymmetricEP`](@ref), using opposite nudges to cancel the leading finite-``\beta``
  error;
- [`HolomorphicEP`](@ref), extracting the objective gradient from equilibria sampled
  around a complex nudging contour; and
- [`ContinuousEP`](@ref), alternating nudged state evolution with local parameter
  updates throughout the learning phase.

For dynamics that do not derive from a scalar energy, [`DynamicalModel`](@ref)
supports:

- [`AsymEP`](@ref), which corrects nudged dynamics using the antisymmetric part of the
  free-state Jacobian; and
- [`DyadicEP`](@ref), which evolves midpoint and adjoint-like difference states.

These interfaces make reciprocal energy-based networks and directed or asymmetric
equilibrium systems available through the same problem, diagnostics, and optimizer
conventions.

## Equilibrium solvers

The built-in [`Relaxation`](@ref) solver provides explicit gradient-flow steps with
absolute and relative convergence tolerances. Optional extensions add:

- [`ODERelaxation`](@ref) with explicit or stiff OrdinaryDiffEq integrators;
- [`SteadyStateRelaxation`](@ref) through SteadyStateDiffEq; and
- [`RootRelaxation`](@ref) through NonlinearSolve.

Free, positive, and negative phases may use independent solver configurations. This
is useful when a precise free equilibrium needs a larger budget than warm-started
nudged phases.

## Models and ecosystem integration

[`ContinuousHopfield`](@ref) constructs multilayer continuous Hopfield networks with
hard-clamped inputs, dense reciprocal couplings, biases, optional recurrent
connections, configurable neuron potentials, and structured parameter trees.

The optional Lux extension supports two complementary boundaries:

- [`lux_energy_model`](@ref) incorporates a Lux layer into a conservative scalar
  energy; and
- [`lux_dynamical_model`](@ref) uses a Lux layer to define an arbitrary vector field
  for AsymEP or Dyadic EP.

Lux parameters retain their native tree structure. Non-trainable model state is kept
explicit and fixed during equilibrium relaxation.

Loading Optimisers.jl activates [`train_step!`](@ref) and
[`continuous_train_step!`](@ref). Descent, momentum, Adam, optimizer chains, and
their state are handled without embedding optimizer policy into the model or EP
algorithm.

## OpenXLA acceleration

The optional Reactant extension compiles fixed-shape training workloads for
accelerator execution. Currently compiled paths include:

- one-sided and symmetric conservative EP;
- tree-valued states and parameters;
- bounded data-dependent convergence through Reactant `@trace while` loops;
- Continuous EP with on-device optimizer updates;
- non-conservative AsymEP and Dyadic EP, including EnzymeMLIR Jacobian operations;
- general Optimisers.jl state, including Adam moments and counters; and
- reusable device inputs and parameters without per-minibatch host transfers.

See [OpenXLA acceleration with Reactant](@ref) for the compilation contract and
[CIFAR-10 with frozen Lux convolutions and a compiled CHN](@ref) for a larger example
that composes two accelerator executables.

## Examples

The documentation includes complete workflows at several scales:

- [Getting started](@ref) develops a quadratic model and its EP gradient;
- [Continuous Hopfield example](@ref) trains a nonlinear CHN on interleaved spirals;
- [MNIST classification example](@ref) trains a ten-class CHN on pooled images;
- [Feedforward MLP with AsymEP](@ref) trains directed Lux dynamics without a scalar
  energy; and
- [CIFAR-10 with frozen Lux convolutions and a compiled CHN](@ref) combines a frozen
  Reactant-compiled convolutional feature extractor with compiled CHN training.

Continue with [EP concepts](@ref) for the estimator definitions and numerical
contracts, or go directly to the [API reference](@ref).
