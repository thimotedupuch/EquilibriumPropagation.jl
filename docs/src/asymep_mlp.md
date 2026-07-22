# Feedforward MLP with AsymEP

Use AsymEP when the dynamics do not come from a scalar energy. This example turns a
directed MLP into an equilibrium model.

## 1. Write the dynamics

The state holds the hidden and output values. Each value moves toward the ordinary
MLP forward pass.

```julia
function mlp_dynamics(state, parameters, input, model_state)
    hidden_width = size(parameters.input_weight, 1)
    hidden = @view state[1:hidden_width, :]
    output = @view state[(hidden_width + 1):end, :]

    hidden_target = tanh.(
        parameters.input_weight * input .+ reshape(parameters.input_bias, :, 1),
    )
    output_target = parameters.output_weight * hidden .+
                    reshape(parameters.output_bias, :, 1)
    return vcat(hidden_target .- hidden, output_target .- output)
end
```

At equilibrium, the state equals the MLP prediction.

## 2. Wrap it as a model

```julia
hidden_width = 16
model = DynamicalModel(
    dynamics=mlp_dynamics,
    cost=(state, target, ps, st) -> begin
        output = @view state[(hidden_width + 1):end, :]
        sum(abs2, output .- target) / (2 * size(state, 2))
    end,
    readout=(state, ps, st) -> @view(state[(hidden_width + 1):end, :]),
    initial_state=(ps, input, st) ->
        zeros(eltype(input), hidden_width + 2, size(input, 2)),
)
```

Parameters may be an array, tuple, or named tuple. The complete script uses Lux to
initialize the named parameter tree.

## 3. Choose AsymEP

```julia
backend = AutoForwardDiff()
algorithm = AsymEP(
    0.15f0,
    Relaxation(dt=0.3f0, maxiters=80, abstol=2f-5);
    state_ad=backend,
    parameter_ad=backend,
)
```

Training then uses the same `train_step!` call as an energy-based model:

```julia
optimizer_state, parameters, stats = train_step!(
    optimizer_state,
    EPProblem(model, parameters, model_state, inputs, targets),
    algorithm,
)
```

## Run the complete script

```bash
julia --project=examples examples/asymep_mlp.jl
```

The script trains on two spirals and prints loss, accuracy, residual, and
convergence.
