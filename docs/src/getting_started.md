# Getting started

## Define a quadratic energy

This complete example has an analytical free equilibrium and is also representative
of the package's low-level interface.

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff # loads DifferentiationInterface's ForwardDiff extension
using LinearAlgebra : dot

function quadratic_energy(s, ps, x, st)
    W = reshape(view(ps, 1:4), 2, 2)
    b = view(ps, 5:6)
    return 0.5f0 * dot(s, s) - dot(s, W, x) - dot(s, b)
end

cost(s, y, ps, st) = 0.5f0 * sum(abs2, s .- y)

model = EPModel(
    energy=quadratic_energy,
    cost=cost,
    readout=(s, ps, st) -> s,
    initial_state=(ps, x, st) -> zeros(eltype(ps), 2),
)

parameters = Float32[0.2, 0.4, -0.1, 0.3, 0, 0]
input = Float32[0.7, -0.4]
target = Float32[-0.2, 0.8]
problem = EPProblem(model, parameters, NamedTuple(), input, target)
```

The four model functions obey these contracts:

- `energy` and `cost` return scalars;
- `initial_state` returns the dynamical state relaxed by the solver;
- `readout` converts an equilibrium state to a prediction; and
- model state remains constant during an equilibrium solve.

## Configure and run EP

Select state and parameter differentiation separately. The implementation package
(`ForwardDiff` here) must be loaded in addition to constructing its ADTypes selector.

```julia
algorithm = EPAlgorithm(
    SymmetricEP(0.05f0),
    Relaxation(dt=0.2f0, maxiters=100, abstol=1f-5);
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoForwardDiff(),
)

phases = solve_phases(problem, algorithm)
gradients = ep_gradient(problem, phases, algorithm)

# Or solve phases, compute gradients, and collect diagnostics together:
gradients, stats = ep_gradient(problem, algorithm)
```

It is possible to inspect the stats of the computation : 
```julia
julia> stats
EPStats
  converged: true
  loss: 0.277001
  gradient norm: 0.958253
  phase displacement: (positive = 0.0354358, negative = 0.0391642)
  phases:
    free: residual=8.39131f-6, iterations=46, energy=-0.029
    positive: residual=9.71078f-6, iterations=35, energy=-0.0158095
    negative: residual=8.1353f-6, iterations=40, energy=-0.0435789
```



Check `stats.converged` before trusting an update. The individual residuals and
iteration counts show which phase needs a smaller step, looser/tighter tolerance, or
larger iteration budget.

## Use different solvers per phase

Nudged phases are often easier to warm-start than the free phase. Configure them
independently without changing the EP protocol:

```julia
algorithm = EPAlgorithm(
    SymmetricEP(0.05f0),
    Relaxation(dt=0.2f0, maxiters=100, abstol=1f-5);
    free_solver=Relaxation(dt=0.15f0, maxiters=150, abstol=1f-6),
    positive_solver=Relaxation(dt=0.2f0, maxiters=75, abstol=1f-5),
    negative_solver=Relaxation(dt=0.2f0, maxiters=75, abstol=1f-5),
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoForwardDiff(),
)
```

## Apply an optimizer update

Loading Optimisers activates the optional training extension:

```julia
using Optimisers

optimizer_state = Optimisers.setup(Optimisers.Adam(1f-3), parameters)
optimizer_state, parameters, stats = train_step!(
    optimizer_state,
    parameters,
    model,
    (input, target),
    algorithm,
)
```

Both optimizer state and model parameters are explicit return values, making a
training loop straightforward to checkpoint.
