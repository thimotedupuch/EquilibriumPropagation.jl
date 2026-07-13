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

struct HopfieldLayout
    ninput::Int
    nhidden::Int
    noutput::Int
end

function unpack(parameters, layout::HopfieldLayout)
    ni, nh, no = layout.ninput, layout.nhidden, layout.noutput
    wx_stop = nh * ni
    wh_stop = wx_stop + no * nh
    bh_stop = wh_stop + nh
    input_weights = reshape(view(parameters, 1:wx_stop), nh, ni)
    hidden_weights = reshape(view(parameters, (wx_stop + 1):wh_stop), no, nh)
    hidden_bias = view(parameters, (wh_stop + 1):bh_stop)
    output_bias = view(parameters, (bh_stop + 1):length(parameters))
    return input_weights, hidden_weights, hidden_bias, output_bias
end

function initialize_parameters(rng::AbstractRNG, layout::HopfieldLayout)
    input_weights = 0.6f0 / sqrt(Float32(layout.ninput)) .* randn(
        rng, Float32, layout.nhidden, layout.ninput,
    )
    hidden_weights = 0.6f0 / sqrt(Float32(layout.nhidden)) .* randn(
        rng, Float32, layout.noutput, layout.nhidden,
    )
    return [vec(input_weights); vec(hidden_weights); zeros(Float32, layout.nhidden + layout.noutput)]
end

function continuous_hopfield_model(layout::HopfieldLayout)
    nh = layout.nhidden
    energy = function (state, parameters, input, model_state)
        input_weights, hidden_weights, hidden_bias, output_bias =
            unpack(parameters, layout)
        hidden = view(state, 1:nh, :)
        output = view(state, (nh + 1):size(state, 1), :)
        hidden_rates = tanh.(hidden)
        output_rates = tanh.(output)
        batch_size = size(input, 2)

        input_drive = input_weights * input .+ hidden_bias
        hidden_drive = hidden_weights * hidden_rates
        return (
            sum(abs2, state) / 2 -
            sum(input_drive .* hidden_rates) -
            sum(hidden_drive .* output_rates) -
            sum(output_bias .* output_rates)
        ) / batch_size
    end

    return EPModel(
        energy=energy,
        cost=(state, target, parameters, model_state) -> begin
            output = view(state, (nh + 1):size(state, 1), :)
            sum(abs2, output .- target) / (2 * size(target, 2))
        end,
        readout=(state, parameters, model_state) ->
            view(state, (nh + 1):size(state, 1), :),
        initial_state=(parameters, input, model_state) ->
            zeros(eltype(parameters), nh + layout.noutput, size(input, 2)),
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
    layout = HopfieldLayout(2, hidden, 2)
    model = continuous_hopfield_model(layout)
    parameters = initialize_parameters(rng, layout)
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
