using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff
using Lux
using Optimisers
using Printf
using Random

"""A feedforward MLP expressed as triangular continuous-time dynamics."""
struct FeedforwardEquilibriumMLP <: Lux.AbstractLuxLayer
    input_width::Int
    hidden_width::Int
    output_width::Int
end

function Lux.initialparameters(rng::AbstractRNG, layer::FeedforwardEquilibriumMLP)
    input_scale = 0.8f0 / sqrt(Float32(layer.input_width))
    output_scale = 0.8f0 / sqrt(Float32(layer.hidden_width))
    return (
        input_weight=input_scale .* randn(
            rng, Float32, layer.hidden_width, layer.input_width,
        ),
        input_bias=zeros(Float32, layer.hidden_width),
        output_weight=output_scale .* randn(
            rng, Float32, layer.output_width, layer.hidden_width,
        ),
        output_bias=zeros(Float32, layer.output_width),
    )
end

Lux.initialstates(rng::AbstractRNG, layer::FeedforwardEquilibriumMLP) = NamedTuple()

function (layer::FeedforwardEquilibriumMLP)(values::Tuple, parameters, model_state)
    state, input = values
    expected = layer.hidden_width + layer.output_width
    size(state, 1) == expected || throw(DimensionMismatch(
        "state has $(size(state, 1)) rows; expected $expected",
    ))
    size(input, 1) == layer.input_width || throw(DimensionMismatch(
        "input has $(size(input, 1)) rows; expected $(layer.input_width)",
    ))
    size(state, 2) == size(input, 2) || throw(DimensionMismatch(
        "state and input batch sizes must match",
    ))
    hidden = @view state[1:layer.hidden_width, :]
    output = @view state[(layer.hidden_width + 1):end, :]
    hidden_target = tanh.(
        parameters.input_weight * input .+ reshape(parameters.input_bias, :, 1),
    )
    output_target = parameters.output_weight * hidden .+
                    reshape(parameters.output_bias, :, 1)
    field = vcat(hidden_target .- hidden, output_target .- output)
    return field, model_state
end

function mlp_spiral_dataset(rng::AbstractRNG, observations::Integer; noise=0.04f0)
    iseven(observations) || throw(ArgumentError("observations must be even"))
    per_class = observations ÷ 2
    inputs = Matrix{Float32}(undef, 2, observations)
    labels = Vector{Int}(undef, observations)
    for class in 1:2, index in 1:per_class
        radius = Float32(index / per_class)
        angle = 1.5f0 * Float32(pi) * radius + Float32(class - 1) * Float32(pi)
        column = (class - 1) * per_class + index
        inputs[:, column] .= radius .* (cos(angle), sin(angle)) .+
                             noise .* randn(rng, Float32, 2)
        labels[column] = class
    end
    targets = -ones(Float32, 2, observations)
    for column in axes(targets, 2)
        targets[labels[column], column] = 1f0
    end
    return inputs, targets, labels
end

function asymep_mlp(hidden::Integer; rng=Xoshiro(7))
    hidden > 0 || throw(ArgumentError("hidden must be positive"))
    layer = FeedforwardEquilibriumMLP(2, hidden, 2)
    parameters, model_state = lux_setup(rng, layer)
    model = lux_dynamical_model(
        layer;
        layer_input=(state, input) -> (state, input),
        cost=(state, target, ps, st) -> begin
            output = @view state[(hidden + 1):end, :]
            sum(abs2, output .- target) / (2 * size(state, 2))
        end,
        readout=(state, ps, st) -> @view(state[(hidden + 1):end, :]),
        initial_state=(ps, input, st) -> zeros(
            eltype(input), hidden + 2, size(input, 2),
        ),
    )
    return model, parameters, model_state
end

asymep_mlp_relaxation(; maxiters=80, abstol=2f-5) =
    Relaxation(dt=0.3f0, maxiters=maxiters, abstol=abstol)

function asymep_mlp_accuracy(model, parameters, model_state, inputs, labels)
    problem = EPProblem(
        model, parameters, model_state, inputs, zeros(Float32, 2, size(inputs, 2)),
    )
    solution = equilibrate(
        problem, FreePhase(), asymep_mlp_relaxation(); state_ad=AutoForwardDiff(),
    )
    scores = predict(solution, problem)
    correct = count(eachindex(labels)) do column
        findmax(@view scores[:, column])[2] == labels[column]
    end
    return correct / length(labels), solution
end

function train_asymep_mlp(
    ; epochs=20, observations=200, batch_size=10, hidden=16, seed=7,
)
    observations % batch_size == 0 || throw(ArgumentError(
        "observations must be divisible by batch_size",
    ))
    rng = Xoshiro(seed)
    inputs, targets, labels = mlp_spiral_dataset(rng, observations)
    model, parameters, model_state = asymep_mlp(hidden; rng)
    backend = AutoForwardDiff()
    algorithm = AsymEP(
        0.15f0,
        asymep_mlp_relaxation();
        state_ad=backend,
        parameter_ad=backend,
    )
    optimizer_state = Optimisers.setup(Optimisers.Adam(5f-3), parameters)
    initial_accuracy, _ = asymep_mlp_accuracy(
        model, parameters, model_state, inputs, labels,
    )
    @printf "feedforward AsymEP MLP: 2-%d-2\n" hidden
    @printf "initial accuracy: %5.1f%%\n" (100 * initial_accuracy)

    for epoch in 1:epochs
        permutation = randperm(rng, observations)
        mean_loss = 0.0
        all_converged = true
        for first in 1:batch_size:observations
            columns = @view permutation[first:(first + batch_size - 1)]
            problem = EPProblem(
                model,
                parameters,
                model_state,
                @view(inputs[:, columns]),
                @view(targets[:, columns]),
            )
            optimizer_state, parameters, stats = train_step!(
                optimizer_state, problem, algorithm,
            )
            mean_loss += stats.loss
            all_converged &= stats.converged
        end
        mean_loss /= observations ÷ batch_size
        if epoch == 1 || epoch % max(1, epochs ÷ 10) == 0 || epoch == epochs
            current_accuracy, solution = asymep_mlp_accuracy(
                model, parameters, model_state, inputs, labels,
            )
            @printf(
                "epoch %3d | loss %.4f | accuracy %5.1f%% | residual %.2e | converged %s\n",
                epoch, mean_loss, 100 * current_accuracy, solution.residual,
                string(all_converged),
            )
        end
    end
    accuracy, _ = asymep_mlp_accuracy(model, parameters, model_state, inputs, labels)
    return (; model, parameters, model_state, inputs, targets, labels, accuracy)
end

function parse_asymep_mlp_options(arguments)
    options = Dict(
        "epochs" => 20,
        "observations" => 200,
        "batch-size" => 10,
        "hidden" => 16,
        "seed" => 7,
    )
    for argument in arguments
        startswith(argument, "--") || throw(ArgumentError("unknown argument: $argument"))
        key_value = split(argument[3:end], '='; limit=2)
        length(key_value) == 2 || throw(ArgumentError("expected --name=value"))
        haskey(options, key_value[1]) || throw(ArgumentError(
            "unknown option: --$(key_value[1])",
        ))
        options[key_value[1]] = parse(Int, key_value[2])
    end
    return options
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_asymep_mlp_options(ARGS)
    result = train_asymep_mlp(
        epochs=options["epochs"],
        observations=options["observations"],
        batch_size=options["batch-size"],
        hidden=options["hidden"],
        seed=options["seed"],
    )
    @printf "final accuracy: %5.1f%%\n" (100 * result.accuracy)
end
