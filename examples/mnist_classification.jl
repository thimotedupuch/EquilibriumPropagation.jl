using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff
using MLDatasets: MNIST
using Optimisers
using Printf
using Random

"""Average-pool 28×28 MNIST images and return a features-by-observations matrix."""
function pool_mnist(images::AbstractArray{<:Real,3}, factor::Integer)
    width, height, observations = size(images)
    width == height || throw(ArgumentError("MNIST images must be square"))
    width % factor == 0 || throw(ArgumentError("pool factor must divide $width"))
    side = width ÷ factor
    inputs = Matrix{Float32}(undef, side * side, observations)
    scale = inv(Float32(factor * factor))

    for observation in 1:observations, column in 1:side, row in 1:side
        rows = ((row - 1) * factor + 1):(row * factor)
        columns = ((column - 1) * factor + 1):(column * factor)
        feature = row + (column - 1) * side
        inputs[feature, observation] = scale * sum(@view images[rows, columns, observation])
    end
    return inputs
end

"""Encode digit labels 0–9 as bipolar targets for the EP nudging cost."""
function mnist_targets(labels::AbstractVector{<:Integer})
    targets = -ones(Float32, 10, length(labels))
    for (column, label) in enumerate(labels)
        0 <= label <= 9 || throw(ArgumentError("expected an MNIST label in 0:9"))
        targets[label + 1, column] = 1f0
    end
    return targets
end

function load_mnist(; train_samples=1_000, test_samples=500, pool=4, seed=7)
    train_images, train_labels = MNIST(:train)[:]
    test_images, test_labels = MNIST(:test)[:]
    train_samples <= length(train_labels) || throw(ArgumentError("too many training samples"))
    test_samples <= length(test_labels) || throw(ArgumentError("too many test samples"))

    rng = Xoshiro(seed)
    train_indices = randperm(rng, length(train_labels))[1:train_samples]
    test_indices = randperm(rng, length(test_labels))[1:test_samples]
    train_inputs = pool_mnist(@view(train_images[:, :, train_indices]), pool)
    test_inputs = pool_mnist(@view(test_images[:, :, test_indices]), pool)
    selected_train_labels = train_labels[train_indices]
    selected_test_labels = test_labels[test_indices]
    return (
        train_inputs,
        mnist_targets(selected_train_labels),
        selected_train_labels,
        test_inputs,
        mnist_targets(selected_test_labels),
        selected_test_labels,
    )
end

"""
Construct a quadratic energy-based classifier.

Images are fixed external inputs. A hidden population and the ten output neurons form
the dynamical state relaxed by EP, and the output free equilibrium contains the class
scores. The shared scalar energy creates reciprocal hidden-output interactions, making
this a genuine continuous Hopfield classifier rather than a linear readout.
"""
function mnist_classifier(ninput::Integer; hidden=64)
    hidden > 0 || throw(ArgumentError("hidden must be positive"))
    weight_initializer = (rng, rows, columns) ->
        0.5f0 / sqrt(Float32(columns)) .* randn(rng, Float32, rows, columns)
    return ContinuousHopfield(
        (ninput, hidden, 10);
        activation=tanh,
        weight_initializer,
    )
end

function mnist_relaxation(batch_size; maxiters=20, abstol=1f-5)
    # The energy and cost are batch means, so their state gradients contain a
    # 1 / batch_size factor. Scale dt to keep the per-neuron step unchanged.
    return Relaxation(dt=0.8f0 * batch_size, maxiters=maxiters, abstol=abstol)
end

function mnist_accuracy(model, parameters, inputs, labels, state_ad; batch_size=20)
    correct = 0
    for first in 1:batch_size:size(inputs, 2)
        last = min(first + batch_size - 1, size(inputs, 2))
        columns = first:last
        batch_inputs = @view inputs[:, columns]
        dummy_targets = zeros(Float32, 10, length(columns))
        problem = EPProblem(model, parameters, NamedTuple(), batch_inputs, dummy_targets)
        solution = equilibrate(
            problem,
            FreePhase(),
            mnist_relaxation(length(columns));
            state_ad=state_ad,
        )
        scores = predict(solution, problem)
        for (local_column, label) in enumerate(@view labels[columns])
            correct += findmax(@view scores[:, local_column])[2] == label + 1
        end
    end
    return correct / length(labels)
end

function parse_mnist_options(arguments)
    options = Dict(
        "epochs" => 10,
        "train-samples" => 1_000,
        "test-samples" => 500,
        "batch-size" => 20,
        "pool" => 4,
        "hidden" => 64,
        "seed" => 7,
    )
    for argument in arguments
        startswith(argument, "--") || throw(ArgumentError("unknown argument: $argument"))
        key_value = split(argument[3:end], '='; limit=2)
        length(key_value) == 2 || throw(ArgumentError("expected --name=value, got $argument"))
        haskey(options, key_value[1]) || throw(ArgumentError("unknown option: --$(key_value[1])"))
        options[key_value[1]] = parse(Int, key_value[2])
    end
    return options
end

function train_mnist(
    ; epochs=10, train_samples=1_000, test_samples=500, batch_size=20, pool=4,
      hidden=64, seed=7,
)
    rng = Xoshiro(seed)
    train_inputs, train_targets, train_labels, test_inputs, _, test_labels =
        load_mnist(; train_samples, test_samples, pool, seed)
    network = mnist_classifier(size(train_inputs, 1); hidden)
    model, parameters = EquilibriumPropagation.setup(rng, network)
    state_ad = AutoForwardDiff()
    algorithm = EPAlgorithm(
        SymmetricEP(0.1f0),
        mnist_relaxation(batch_size);
        state_ad,
        parameter_ad=AutoForwardDiff(),
    )
    optimizer_state = Optimisers.setup(Optimisers.Adam(1f-2), parameters)

    initial_accuracy = mnist_accuracy(
        model, parameters, test_inputs, test_labels, state_ad; batch_size,
    )
    @printf "MNIST Hopfield classifier: %d-%d-10\n" size(train_inputs, 1) hidden
    @printf "initial test accuracy: %5.1f%%\n" (100 * initial_accuracy)

    for epoch in 1:epochs
        permutation = randperm(rng, train_samples)
        total_loss = 0.0
        batches = 0
        all_converged = true
        for first in 1:batch_size:train_samples
            last = min(first + batch_size - 1, train_samples)
            columns = @view permutation[first:last]
            inputs = @view train_inputs[:, columns]
            targets = @view train_targets[:, columns]
            batch_algorithm = length(columns) == batch_size ? algorithm : EPAlgorithm(
                algorithm.protocol,
                mnist_relaxation(length(columns));
                state_ad=algorithm.state_ad,
                parameter_ad=algorithm.parameter_ad,
            )
            optimizer_state, parameters, stats = train_step!(
                optimizer_state,
                parameters,
                model,
                (inputs, targets),
                batch_algorithm,
            )
            total_loss += stats.loss
            batches += 1
            all_converged &= stats.converged
        end

        test_accuracy = mnist_accuracy(
            model, parameters, test_inputs, test_labels, state_ad; batch_size,
        )
        @printf(
            "epoch %2d | loss %.4f | test accuracy %5.1f%% | converged %s\n",
            epoch,
            total_loss / batches,
            100 * test_accuracy,
            string(all_converged),
        )
    end

    final_accuracy = mnist_accuracy(
        model, parameters, test_inputs, test_labels, state_ad; batch_size,
    )
    return (; model, parameters, network, accuracy=final_accuracy)
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_mnist_options(ARGS)
    result = train_mnist(
        epochs=options["epochs"],
        train_samples=options["train-samples"],
        test_samples=options["test-samples"],
        batch_size=options["batch-size"],
        pool=options["pool"],
        hidden=options["hidden"],
        seed=options["seed"],
    )
    @printf "final test accuracy: %5.1f%%\n" (100 * result.accuracy)
end
