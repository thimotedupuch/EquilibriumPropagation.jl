# Lux integration

The optional Lux extension adapts explicit Lux layers without merging Lux's model
state into the EP dynamical state. Load Lux alongside EquilibriumPropagation to
activate it:

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using Lux
using Random

rng = Xoshiro(42)
layer = Lux.Dense(2 => 2, identity)
parameters, model_state = lux_setup(rng, layer)
```

`lux_setup` delegates parameter and state initialization to `Lux.setup`, then applies
`Lux.testmode` to the state. During equilibrium evaluation, the adapter calls
`Lux.apply(layer, layer_input, parameters, model_state)` and requires the returned
state to equal the supplied state. A layer that updates BatchNorm statistics, dropout
state, counters, or any other non-trainable state is rejected because it would change
the vector field or energy while the solver is relaxing it.

## Vector-field layers

Use `lux_dynamical_model` when the Lux layer computes all or part of `ds/dt`:

```julia
model = lux_dynamical_model(
    layer;
    cost=(state, target, ps, st) -> sum(abs2, state .- target) / 2,
    readout=(state, ps, st) -> state,
    initial_state=(ps, input, st) -> zeros(2),
    layer_input=(state, input) -> state,
    dynamics_output=(output, state, input) -> output,
)

problem = EPProblem(model, parameters, model_state, nothing, target)
algorithm = DyadicEP(
    1e-3,
    Relaxation(dt=0.1, maxiters=500, abstol=1e-8);
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoForwardDiff(),
)
gradient, stats = ep_gradient(problem, algorithm)
```

`layer_input` explicitly packs the EP state and external input into the value expected
by the Lux layer. `dynamics_output` converts the ordinary Lux output into a vector
field with the same shape as the EP state. The adapter works with AsymEP and Dyadic EP;
the Lux parameter tree remains directly compatible with Optimisers.jl.

## Scalar-energy layers

Use `lux_energy_model` when a Lux layer contributes to a conservative scalar energy:

```julia
energy_model = lux_energy_model(
    Lux.Dense(2 => 1; use_bias=false);
    energy_output=(output, state, input) ->
        sum(state .^ 2) / 2 - only(output),
    cost=(state, target, ps, st) -> sum((state .- target) .^ 2) / 2,
    readout=(state, ps, st) -> state,
    initial_state=(ps, input, st) -> zeros(2),
)
```

`energy_output` must return a scalar. The resulting ordinary `EPModel` supports
classical, Continuous, and Holomorphic EP. For Holomorphic EP, the Lux layer,
activation, `energy_output`, and cost must all satisfy the complex-valued contract in
the [Holomorphic EP guide](holomorphic_ep.md).

This boundary preserves Lux's conventions:

- the Lux layer describes ordinary feedforward evaluation;
- the `EPProblem` state is the value evolved by equilibrium relaxation;
- Lux parameters are trainable and differentiated by the selected AD backend;
- Lux model state is non-trainable and frozen; and
- method-specific adjoint or doubled states remain owned by AsymEP or Dyadic EP.
