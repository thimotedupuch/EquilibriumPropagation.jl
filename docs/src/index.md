# EquilibriumPropagation.jl

EquilibriumPropagation.jl is a small Julia toolkit for equilibrium propagation.

- Build energy-based models or provide your own dynamics.
- Choose classical, continuous, holomorphic, or non-conservative EP.
- Train with Optimisers.jl and compile fixed-shape workloads with Reactant.

## Minimum working example

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff
using Random

rng = Xoshiro(1)
model, parameters = EquilibriumPropagation.setup(
    rng,
    ContinuousHopfield((2, 3, 1)),
)

problem = EPProblem(
    model,
    parameters,
    NamedTuple(),
    Float32[0.2, -0.1],
    Float32[0.5],
)

backend = AutoForwardDiff()
algorithm = EPAlgorithm(
    SymmetricEP(0.05f0),
    Relaxation(dt=0.2f0, maxiters=500, abstol=1f-5);
    state_ad=backend,
    parameter_ad=backend,
)

gradient, stats = ep_gradient(problem, algorithm)
```

That is the full EP workflow. `gradient` matches the parameter tree. Check
`stats.converged` before using it.

## Update the model

Optimisers.jl adds a one-call training step:

```julia
using Optimisers

optimizer = Optimisers.setup(Optimisers.Adam(1f-3), parameters)
optimizer, parameters, stats = train_step!(
    optimizer,
    parameters,
    model,
    (problem.input, problem.target),
    algorithm,
)
```

Start with [Getting started](@ref) for the low-level model interface. See the
[Continuous Hopfield example](@ref) for a training loop. The [API reference](@ref)
contains the full package.
