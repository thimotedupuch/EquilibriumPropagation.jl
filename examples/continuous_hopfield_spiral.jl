using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff
using LinearAlgebra: norm
using Optimisers
using Printf
using Random

"""Generate two interleaved planar spirals and their bipolar class targets."""
function spiral_dataset(rng::AbstractRNG, n::Integer; noise=0.035f0)
    iseven(n) || throw(ArgumentError("the number of points must be even"))
    per_class = n ÷ 2
    inputs = Matrix{Float32}(undef, 2, n)
    labels = Vector{Int}(undef, n)

    for class in 1:2, index in 1:per_class
        radius = Float32(index / per_class)
        angle = 1.5f0 * Float32(pi) * radius + Float32(class - 1) * Float32(pi)
        column = (class - 1) * per_class + index
        inputs[:, column] .= radius .* (cos(angle), sin(angle)) .+
                             noise .* randn(rng, Float32, 2)
        labels[column] = class
    end

    targets = -ones(Float32, 2, n)
    for column in axes(targets, 2)
        targets[labels[column], column] = 1
    end
    return inputs, targets, labels
end

function spiral_network(hidden::Integer)
    weight_initializer = (rng, rows, columns) ->
        0.6f0 / sqrt(Float32(columns)) .* randn(rng, Float32, rows, columns)
    return ContinuousHopfield(
        (2, hidden, 2);
        activation=tanh,
        potential=QuadraticPotential(),
        input_clamp=HardClamp(),
        output_cost=BipolarSquaredError(),
        bias=true,
        recurrent=false,
        weight_initializer,
    )
end

function relaxation_for(batch_size; maxiters=50, abstol=5f-4)
    # Both energy terms are batch means. Scaling dt by the batch size keeps the
    # effective per-neuron Euler step independent of minibatch size.
    return Relaxation(
        dt=0.35f0 * batch_size,
        maxiters=maxiters,
        abstol=abstol,
    )
end

function accuracy(model, parameters, inputs, labels, state_ad)
    problem = EPProblem(model, parameters, NamedTuple(), inputs, zeros(Float32, 2, size(inputs, 2)))
    solution = equilibrate(
        problem,
        FreePhase(),
        relaxation_for(size(inputs, 2); maxiters=80, abstol=2f-4);
        state_ad=state_ad,
    )
    scores = predict(solution, problem)
    correct = count(eachindex(labels)) do column
        findmax(view(scores, :, column))[2] == labels[column]
    end
    return correct / length(labels), solution.residual
end

function parse_options(arguments)
    options = Dict(
        "epochs" => 30,
        "points" => 400,
        "batch-size" => 20,
        "hidden" => 32,
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

function train_spiral(; epochs=30, points=400, batch_size=20, hidden=32, seed=7)
    points % batch_size == 0 || throw(ArgumentError("points must be divisible by batch_size"))
    rng = Xoshiro(seed)
    inputs, targets, labels = spiral_dataset(rng, points)
    network = spiral_network(hidden)
    model, parameters = EquilibriumPropagation.setup(rng, network)
    state_ad = AutoForwardDiff()
    algorithm = EPAlgorithm(
        SymmetricEP(0.15f0),
        relaxation_for(batch_size);
        state_ad=state_ad,
        parameter_ad=AutoForwardDiff(),
    )
    optimizer_state = Optimisers.setup(Optimisers.Adam(1f-2), parameters)
    report_every = max(1, epochs ÷ 10)

    initial_accuracy, _ = accuracy(model, parameters, inputs, labels, state_ad)
    @printf "continuous Hopfield network: %d-%d-%d\n" 2 hidden 2
    @printf "initial accuracy: %5.1f%%\n" (100 * initial_accuracy)

    for epoch in 1:epochs
        permutation = randperm(rng, points)
        total_loss = 0.0
        all_converged = true
        for offset in 1:batch_size:points
            columns = view(permutation, offset:(offset + batch_size - 1))
            batch = (inputs[:, columns], targets[:, columns])
            optimizer_state, parameters, stats = train_step!(
                optimizer_state,
                parameters,
                model,
                batch,
                algorithm,
            )
            total_loss += stats.loss
            all_converged &= stats.converged
        end

        if epoch == 1 || epoch % report_every == 0 || epoch == epochs
            current_accuracy, residual = accuracy(model, parameters, inputs, labels, state_ad)
            mean_loss = total_loss / (points ÷ batch_size)
            @printf(
                "epoch %3d | loss %.4f | accuracy %5.1f%% | residual %.2e | converged %s\n",
                epoch,
                mean_loss,
                100 * current_accuracy,
                residual,
                string(all_converged),
            )
        end
    end

    final_accuracy, _ = accuracy(model, parameters, inputs, labels, state_ad)
    return (; model, parameters, inputs, targets, labels, accuracy=final_accuracy)
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_options(ARGS)
    result = train_spiral(
        epochs=options["epochs"],
        points=options["points"],
        batch_size=options["batch-size"],
        hidden=options["hidden"],
        seed=options["seed"],
    )
    @printf "final accuracy: %5.1f%%\n" (100 * result.accuracy)
end
