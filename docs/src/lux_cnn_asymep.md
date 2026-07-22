# Train a Lux CNN with AsymEP

This example trains a small Lux CNN on Fashion-MNIST. Only its ten output logits are
equilibrium variables.

## 1. Define the predictor

```julia
layer = Lux.Chain(
    Lux.Conv((3, 3), 1 => 8, relu; pad=Lux.SamePad()),
    Lux.MaxPool((2, 2)),
    Lux.Conv((3, 3), 8 => 16, relu; pad=Lux.SamePad()),
    Lux.GlobalMeanPool(),
    Lux.FlattenLayer(),
    Lux.Dense(16 => 10),
)
```

## 2. Make its output an equilibrium

For a predictor `f(x)`, use the dynamics `f(x) - state`. The free equilibrium is
therefore `state == f(x)`.

```julia
model = lux_dynamical_model(
    layer;
    layer_input=(state, input) -> reshape(input, 28, 28, 1, size(input, 2)),
    dynamics_output=(logits, state, input) -> logits .- state,
    cost=(state, target, ps, st) ->
        sum(abs2, state .- target) / (2 * size(state, 2)),
    readout=(state, ps, st) -> state,
    initial_state=(ps, input, st) -> zeros(eltype(input), 10, size(input, 2)),
)
```

Initialize the Lux parameters and fixed model state:

```julia
parameters, model_state = lux_setup(rng, layer)
```

## 3. Choose the differentiation backends

The equilibrium state is small, so forward mode works well for it. The CNN has many
parameters, so the example uses reverse-mode Enzyme for parameters.

```julia
algorithm = AsymEP(
    0.1f0,
    Relaxation(dt=1f0, maxiters=12, abstol=2f-5);
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoEnzyme(function_annotation=Enzyme.Duplicated),
)
```

Now create an `EPProblem` and call `train_step!` as usual.

## Run the complete script

```bash
julia --project=examples examples/lux_cnn_asymep.jl --epochs=1
```

Use `--method=dyadic` to replace AsymEP with Dyadic EP. The first Enzyme call may
take longer because it compiles the reverse pass.
