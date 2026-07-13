# Non-conservative EP

Classical EP assumes that inference follows the gradient of a scalar energy.
`DynamicalModel` makes the alternative explicit: its `dynamics` callable returns an
arbitrary differentiable vector field and can therefore represent asymmetric or
feedforward interactions.

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation

J = [-2.0 1.0; 0.0 -1.0]
model = DynamicalModel(
    dynamics=(state, parameters, input, model_state) -> J * state + parameters,
    cost=(state, target, parameters, model_state) ->
        sum(abs2, state .- target) / 2,
    readout=(state, parameters, model_state) -> state,
    initial_state=(parameters, input, model_state) -> zeros(2),
)
problem = EPProblem(model, [0.3, -0.2], NamedTuple(), nothing, [-0.4, 0.7])
solver = Relaxation(dt=0.1, maxiters=1000, abstol=1e-10)
backend = AutoForwardDiff()

algorithm = AsymEP(
    1e-3, solver; state_ad=backend, parameter_ad=backend,
)
gradient, stats = ep_gradient(problem, algorithm)
```

## AsymEP

Let `F(s, θ)` be the vector field, `s₀` its free equilibrium, and
`A = (J_F - J_Fᵀ)/2` the antisymmetric part of its Jacobian at `s₀`. For opposite
nudging strengths, AsymEP relaxes

```math
\dot{s} = F(s,\theta) - \beta\nabla_s C(s,\theta)
          - 2A(s-s_0).
```

The correction changes the linearized nudged dynamics from `J_F` to `J_Fᵀ`. The
central displacement therefore supplies the equilibrium adjoint needed for the exact
cost gradient in the limit of small nudging.

## Dyadic EP

Dyadic EP uses two coupled states, represented internally by their midpoint `m` and
difference `d`:

```math
\dot m = F(m,\theta), \qquad
\dot d = J_F(m,\theta)^\mathsf{T}d - \beta\nabla_m C(m,\theta).
```

Starting from `m=s₀` and `d=0`, the midpoint retains the original inference dynamics
while `d/β` relaxes to the adjoint signal. Select it by changing only the algorithm:

```julia
algorithm = DyadicEP(
    1e-3, solver; state_ad=backend, parameter_ad=backend,
)
gradient, stats = ep_gradient(problem, algorithm)
```

Both algorithms include any explicit parameter derivative of the cost, work with
array or tuple/named-tuple parameter trees, and integrate with `train_step!` after
loading Optimisers.jl. Their current state space is restricted to numeric arrays and
their solver to fixed-step `Relaxation`.

The algorithms follow Scurria et al., *Equilibrium Propagation for Non-Conservative
Systems*, arXiv:2602.03670 (2026).
