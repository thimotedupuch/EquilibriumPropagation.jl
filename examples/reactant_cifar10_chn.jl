using EquilibriumPropagation
using Enzyme
using Lux
using MLDatasets: CIFAR10
using Optimisers
using Printf
using Random
using Reactant

struct FrozenLuxFeatures{L}
    layer::L
end

function (extractor::FrozenLuxFeatures)(images, parameters, state)
    features, _ = Lux.apply(extractor.layer, images, parameters, state)
    # FlattenLayer exposes a lazy ReshapedArray. The broadcast materializes an
    # ordinary device tensor that can be passed directly to the EP executable.
    return features .+ zero(eltype(features))
end

function cifar_feature_extractor(channels::Integer)
    channels >= 4 || throw(ArgumentError("channels must be at least four"))
    first_channels = max(4, channels ÷ 2)
    return Lux.Chain(
        Lux.WrappedFunction(x -> 2f0 .* x .- 1f0),
        Lux.Conv((5, 5), 3 => first_channels, tanh; stride=2, pad=Lux.SamePad()),
        Lux.MeanPool((2, 2)),
        Lux.Conv(
            (3, 3), first_channels => channels, tanh;
            stride=2, pad=Lux.SamePad(),
        ),
        Lux.GlobalMeanPool(),
        Lux.FlattenLayer(),
    )
end

function bipolar_cifar_targets(labels)
    all(label -> 1 <= label <= 10, labels) || throw(ArgumentError(
        "labels must use Julia's 1:10 class indexing",
    ))
    targets = -ones(Float32, 10, length(labels))
    for (column, label) in enumerate(labels)
        targets[label, column] = 1f0
    end
    return targets
end

function cifar_chn(rng, feature_width::Integer, hidden::Integer)
    initializer = (rng, rows, columns) ->
        0.45f0 / sqrt(Float32(columns)) .* randn(rng, Float32, rows, columns)
    specification = ContinuousHopfield(
        (feature_width, hidden, 10);
        activation=tanh,
        potential=QuadraticPotential(),
        input_clamp=HardClamp(),
        output_cost=BipolarSquaredError(),
        bias=true,
        recurrent=false,
        weight_initializer=initializer,
    )
    return EquilibriumPropagation.setup(rng, specification)
end

function compile_frozen_features(
    rng, sample_images, channels; backend="cpu",
)
    layer = cifar_feature_extractor(channels)
    parameters, state = Lux.setup(rng, layer)
    state = Lux.testmode(state)
    Reactant.set_default_backend(backend)
    device_arguments = (
        Reactant.to_rarray(sample_images; track_numbers=Number),
        Reactant.to_rarray(parameters; track_numbers=Number),
        Reactant.to_rarray(state; track_numbers=Number),
    )
    executable = Reactant.compile(FrozenLuxFeatures(layer), device_arguments)
    return executable, device_arguments[2], device_arguments[3]
end

function extract_cifar_batches(
    executable, parameters, state, images, batch_size,
)
    size(images, 4) % batch_size == 0 || throw(ArgumentError(
        "image count must be divisible by batch_size",
    ))
    return map(1:batch_size:size(images, 4)) do first_index
        columns = first_index:(first_index + batch_size - 1)
        device_images = Reactant.to_rarray(images[:, :, :, columns]; track_numbers=Number)
        return executable(device_images, parameters, state)
    end
end

function feature_problems(model, parameters, feature_batches, targets, batch_size)
    return map(enumerate(feature_batches)) do (index, features)
        columns = ((index - 1) * batch_size + 1):(index * batch_size)
        EPProblem(
            model,
            parameters,
            NamedTuple(),
            EPBatch(features, targets[:, columns]),
        )
    end
end

function output_scores(free_state, hidden)
    host_state = Array(free_state)
    return @view host_state[(hidden + 1):(hidden + 10), :]
end

function batch_correct(scores, labels)
    return count(eachindex(labels)) do column
        findmax(@view scores[:, column])[2] == labels[column]
    end
end

function compiled_accuracy(
    executable, parameters, optimizer_state, inputs, labels, hidden, batch_size,
)
    correct = 0
    for (index, prepared) in enumerate(inputs)
        result = executable(
            parameters,
            prepared.model_state,
            prepared.batch,
            prepared.initial_state,
            optimizer_state,
        )
        columns = ((index - 1) * batch_size + 1):(index * batch_size)
        correct += batch_correct(output_scores(result.free_state, hidden), labels[columns])
    end
    return correct / length(labels)
end

function train_cifar10_chn(
    ; backend="cpu", epochs=8, train_size=4096, test_size=1024,
      batch_size=64, channels=32, hidden=128, seed=17,
)
    train_size % batch_size == 0 || throw(ArgumentError(
        "train_size must be divisible by batch_size",
    ))
    test_size % batch_size == 0 || throw(ArgumentError(
        "test_size must be divisible by batch_size",
    ))
    rng = Xoshiro(seed)
    train_data = CIFAR10(split=:train, Tx=Float32)
    test_data = CIFAR10(split=:test, Tx=Float32)
    train_images = train_data.features[:, :, :, 1:train_size]
    test_images = test_data.features[:, :, :, 1:test_size]
    # CIFAR's binary files use labels 0:9; Julia score rows use 1:10.
    train_labels = train_data.targets[1:train_size] .+ 1
    test_labels = test_data.targets[1:test_size] .+ 1
    train_targets = bipolar_cifar_targets(train_labels)
    test_targets = bipolar_cifar_targets(test_labels)

    @printf "backend: %s | CIFAR-10: %d train / %d test\n" backend train_size test_size
    feature_seconds = @elapsed begin
        feature_executable, feature_parameters, feature_state =
            compile_frozen_features(rng, train_images[:, :, :, 1:batch_size], channels; backend)
        train_features = extract_cifar_batches(
            feature_executable, feature_parameters, feature_state,
            train_images, batch_size,
        )
        test_features = extract_cifar_batches(
            feature_executable, feature_parameters, feature_state,
            test_images, batch_size,
        )
    end
    @printf "compiled and cached frozen convolutional features in %.2f s\n" feature_seconds

    model, parameters = cifar_chn(rng, channels, hidden)
    train_problems = feature_problems(
        model, parameters, train_features, train_targets, batch_size,
    )
    test_problems = feature_problems(
        model, parameters, test_features, test_targets, batch_size,
    )
    train_inputs = map(problem -> reactant_inputs(problem; backend), train_problems)
    test_inputs = map(problem -> reactant_inputs(problem; backend), test_problems)

    algorithm = ReactantEP(
        SymmetricEP(0.12f0);
        dt=0.28f0 * batch_size,
        free_steps=30,
        nudged_steps=15,
        abstol=4f-3,
        reltol=1f-4,
    )
    compilation_seconds = @elapsed step = compile_reactant(
        first(train_problems), algorithm, Optimisers.Adam(2f-3); backend,
    )
    @printf "compiled CHN relaxation, EP gradients, and Adam in %.2f s\n" compilation_seconds

    initial = step.inputs
    device_parameters = initial.parameters
    optimizer_state = initial.optimizer_state
    for epoch in 1:epochs
        order = randperm(rng, length(train_inputs))
        mean_loss = 0f0
        converged_batches = 0
        for index in order
            prepared = train_inputs[index]
            result = step(
                device_parameters,
                prepared.model_state,
                prepared.batch,
                prepared.initial_state,
                optimizer_state,
            )
            device_parameters = result.parameters
            optimizer_state = result.optimizer_state
            mean_loss += Float32(result.loss)
            converged_batches += Bool(result.converged)
        end
        mean_loss /= length(train_inputs)
        if epoch == 1 || epoch == epochs || epoch % max(1, epochs ÷ 4) == 0
            test_accuracy = compiled_accuracy(
                step, device_parameters, optimizer_state, test_inputs,
                test_labels, hidden, batch_size,
            )
            @printf(
                "epoch %2d | loss %.4f | test accuracy %5.1f%% | converged %d/%d\n",
                epoch, mean_loss, 100 * test_accuracy,
                converged_batches, length(train_inputs),
            )
        end
    end
    accuracy = compiled_accuracy(
        step, device_parameters, optimizer_state, test_inputs,
        test_labels, hidden, batch_size,
    )
    return (;
        parameters=device_parameters, optimizer_state, accuracy,
        feature_executable, step,
    )
end

function parse_cifar_options(arguments)
    options = Dict(
        "backend" => "cpu", "epochs" => "8", "train-size" => "4096",
        "test-size" => "1024", "batch-size" => "64", "channels" => "32",
        "hidden" => "128", "seed" => "17",
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
    options = parse_cifar_options(ARGS)
    result = train_cifar10_chn(
        backend=options["backend"],
        epochs=parse(Int, options["epochs"]),
        train_size=parse(Int, options["train-size"]),
        test_size=parse(Int, options["test-size"]),
        batch_size=parse(Int, options["batch-size"]),
        channels=parse(Int, options["channels"]),
        hidden=parse(Int, options["hidden"]),
        seed=parse(Int, options["seed"]),
    )
    @printf "final test accuracy: %5.1f%%\n" (100 * result.accuracy)
end
