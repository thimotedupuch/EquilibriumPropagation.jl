using ADTypes: AutoEnzyme, AutoForwardDiff
using Enzyme
using EquilibriumPropagation
using ForwardDiff
using Lux
using MLDatasets: FashionMNIST
using Optimisers
using Printf
using Random

function small_vgg(; channels=(8, 16))
    first_channels, second_channels = channels
    return Lux.Chain(
        Lux.Conv((3, 3), 1 => first_channels, relu; pad=Lux.SamePad()),
        Lux.Conv(
            (3, 3), first_channels => first_channels, relu;
            pad=Lux.SamePad(),
        ),
        Lux.MaxPool((2, 2)),
        Lux.Conv(
            (3, 3), first_channels => second_channels, relu;
            pad=Lux.SamePad(),
        ),
        Lux.Conv(
            (3, 3), second_channels => second_channels, relu;
            pad=Lux.SamePad(),
        ),
        Lux.GlobalMeanPool(),
        Lux.FlattenLayer(),
        Lux.Dense(second_channels => 10),
    )
end

function output_equilibrium_model(layer::Lux.AbstractLuxLayer)
    return lux_dynamical_model(
        layer;
        layer_input=(state, input) -> reshape(input, 28, 28, 1, size(input, 2)),
        dynamics_output=(logits, state, input) -> logits .- state,
        cost=(state, target, ps, st) ->
            sum(abs2, state .- target) / (2 * size(state, 2)),
        readout=(state, ps, st) -> state,
        initial_state=(ps, input, st) -> zeros(eltype(input), 10, size(input, 2)),
    )
end

function fashion_targets(labels)
    targets = -ones(Float32, 10, length(labels))
    for (column, label) in enumerate(labels)
        targets[label, column] = 1f0
    end
    return targets
end

cnn_relaxation(; maxiters=12, abstol=2f-5) =
    Relaxation(dt=1f0, maxiters=maxiters, abstol=abstol)

function cnn_asymep_algorithm(method::Symbol)
    solver = cnn_relaxation()
    state_ad = AutoForwardDiff()
    # Lux parameter trees contain mutable arrays.  The duplicated-function
    # annotation lets Enzyme differentiate the packed parameter objective.
    parameter_ad = AutoEnzyme(function_annotation=Enzyme.Duplicated)
    if method === :asym
        return AsymEP(
            0.1f0, solver; state_ad, parameter_ad,
        )
    elseif method === :dyadic
        return DyadicEP(
            0.1f0, solver; state_ad, parameter_ad,
        )
    end
    throw(ArgumentError("method must be :asym or :dyadic"))
end

function cnn_scores(layer, parameters, model_state, inputs)
    images = reshape(inputs, 28, 28, 1, size(inputs, 2))
    scores, new_state = Lux.apply(layer, images, parameters, model_state)
    isequal(new_state, model_state) || throw(ArgumentError(
        "the tutorial requires a state-invariant Lux network",
    ))
    return scores
end

function classification_accuracy(scores, labels)
    correct = count(eachindex(labels)) do column
        findmax(@view scores[:, column])[2] == labels[column]
    end
    return correct / length(labels)
end

function train_lux_cnn_asymep(
    ; method=:asym, epochs=5, train_size=1000, test_size=500,
      batch_size=20, channels=(8, 16), seed=23,
)
    train_size % batch_size == 0 || throw(ArgumentError(
        "train_size must be divisible by batch_size",
    ))
    rng = Xoshiro(seed)
    train_data = FashionMNIST(split=:train, Tx=Float32)
    test_data = FashionMNIST(split=:test, Tx=Float32)
    train_inputs = reshape(train_data.features[:, :, 1:train_size], 28 * 28, train_size)
    test_inputs = reshape(test_data.features[:, :, 1:test_size], 28 * 28, test_size)
    train_labels = train_data.targets[1:train_size] .+ 1
    test_labels = test_data.targets[1:test_size] .+ 1
    train_targets = fashion_targets(train_labels)

    layer = small_vgg(; channels)
    parameters, model_state = lux_setup(rng, layer)
    model = output_equilibrium_model(layer)
    algorithm = cnn_asymep_algorithm(method)
    optimizer_state = Optimisers.setup(Optimisers.Adam(2f-3), parameters)

    initial_accuracy = classification_accuracy(
        cnn_scores(layer, parameters, model_state, test_inputs), test_labels,
    )
    @printf(
        "Lux VGG-style CNN with %s: %d train / %d test | initial accuracy %5.1f%%\n",
        string(method), train_size, test_size, 100 * initial_accuracy,
    )

    for epoch in 1:epochs
        permutation = randperm(rng, train_size)
        mean_loss = 0f0
        converged_batches = 0
        for first_index in 1:batch_size:train_size
            columns = @view permutation[first_index:(first_index + batch_size - 1)]
            problem = EPProblem(
                model,
                parameters,
                model_state,
                @view(train_inputs[:, columns]),
                @view(train_targets[:, columns]),
            )
            optimizer_state, parameters, stats = train_step!(
                optimizer_state, problem, algorithm,
            )
            mean_loss += Float32(stats.loss)
            converged_batches += stats.converged
        end
        mean_loss /= train_size ÷ batch_size
        if epoch == 1 || epoch == epochs || epoch % max(1, epochs ÷ 5) == 0
            accuracy = classification_accuracy(
                cnn_scores(layer, parameters, model_state, test_inputs), test_labels,
            )
            @printf(
                "epoch %2d | loss %.4f | test accuracy %5.1f%% | converged %d/%d\n",
                epoch, mean_loss, 100 * accuracy,
                converged_batches, train_size ÷ batch_size,
            )
        end
    end

    accuracy = classification_accuracy(
        cnn_scores(layer, parameters, model_state, test_inputs), test_labels,
    )
    return (; layer, model, parameters, model_state, accuracy, algorithm)
end

function parse_cnn_asymep_options(arguments)
    options = Dict(
        "method" => "asym", "epochs" => "5", "train-size" => "1000",
        "test-size" => "500", "batch-size" => "20", "channels" => "8,16",
        "seed" => "23",
    )
    for argument in arguments
        startswith(argument, "--") || throw(ArgumentError("unknown argument: $argument"))
        key_value = split(argument[3:end], '='; limit=2)
        length(key_value) == 2 || throw(ArgumentError("expected --name=value"))
        haskey(options, key_value[1]) || throw(ArgumentError(
            "unknown option: --$(key_value[1])",
        ))
        options[key_value[1]] = key_value[2]
    end
    return options
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_cnn_asymep_options(ARGS)
    channel_values = parse.(Int, split(options["channels"], ','))
    length(channel_values) == 2 || throw(ArgumentError(
        "--channels must contain two comma-separated widths",
    ))
    result = train_lux_cnn_asymep(
        method=Symbol(options["method"]),
        epochs=parse(Int, options["epochs"]),
        train_size=parse(Int, options["train-size"]),
        test_size=parse(Int, options["test-size"]),
        batch_size=parse(Int, options["batch-size"]),
        channels=Tuple(channel_values),
        seed=parse(Int, options["seed"]),
    )
    @printf "final test accuracy: %5.1f%%\n" (100 * result.accuracy)
end
