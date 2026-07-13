using EquilibriumPropagation
using Enzyme
using Printf
using Random
using Reactant

reactant_linear_energy(state, weights, input, model_state) =
    (sum(abs2, state) / 2 - sum(state .* (weights * input))) / size(input, 2)

reactant_linear_cost(state, target, weights, model_state) =
    sum(abs2, state .- target) / (2 * size(target, 2))

reactant_linear_readout(state, weights, model_state) = state
reactant_linear_initial(weights, input, model_state) = zero(weights * input)

function reactant_linear_dataset(rng::AbstractRNG, observations::Integer)
    iseven(observations) || throw(ArgumentError("observations must be even"))
    half = observations ÷ 2
    input = 0.35f0 .* randn(rng, Float32, 2, observations)
    input[:, 1:half] .+= reshape(Float32[-0.8, -0.5], :, 1)
    input[:, (half + 1):end] .+= reshape(Float32[0.8, 0.5], :, 1)
    target = -ones(Float32, 2, observations)
    target[1, 1:half] .= 1
    target[2, (half + 1):end] .= 1
    labels = vcat(fill(1, half), fill(2, half))
    return input, target, labels
end

function reactant_linear_accuracy(parameters, input, labels)
    scores = Array(parameters) * input
    correct = count(eachindex(labels)) do column
        findmax(@view scores[:, column])[2] == labels[column]
    end
    return correct / length(labels)
end

function train_reactant_ep(
    ; backend="cpu", epochs=12, observations=256, batch_size=32, seed=7,
)
    observations % batch_size == 0 || throw(ArgumentError(
        "observations must be divisible by batch_size (compiled shapes are static)",
    ))
    rng = Xoshiro(seed)
    input, target, labels = reactant_linear_dataset(rng, observations)
    parameters = 0.1f0 .* randn(rng, Float32, 2, 2)
    model = EPModel(
        energy=reactant_linear_energy,
        cost=reactant_linear_cost,
        readout=reactant_linear_readout,
        initial_state=reactant_linear_initial,
    )

    ranges = collect(Iterators.partition(axes(input, 2), batch_size))
    host_problems = map(ranges) do columns
        EPProblem(
            model,
            parameters,
            NamedTuple(),
            EPBatch(input[:, columns], target[:, columns]),
        )
    end
    algorithm = ReactantEP(
        SymmetricEP(0.15f0);
        # Both energy terms are batch means, so this preserves an O(1) Euler step.
        dt=0.55f0 * batch_size,
        free_steps=8,
        nudged_steps=6,
        learning_rate=0.08f0,
    )

    @printf "Reactant/OpenXLA backend: %s\n" backend
    compilation_seconds = @elapsed executable = compile_reactant(
        first(host_problems), algorithm; backend,
    )
    @printf "compiled fixed-shape EP step in %.2f s\n" compilation_seconds

    # Transfer every fixed-shape minibatch once. The compiled executable and device
    # parameters are then reused without recompilation throughout training.
    device_inputs = map(problem -> reactant_inputs(problem; backend), host_problems)
    device_parameters = first(device_inputs).parameters
    initial_accuracy = reactant_linear_accuracy(device_parameters, input, labels)
    @printf "initial accuracy: %5.1f%%\n" (100 * initial_accuracy)

    for epoch in 1:epochs
        mean_loss = 0f0
        max_residual = 0f0
        for inputs in device_inputs
            result = executable(
                device_parameters,
                inputs.model_state,
                inputs.batch,
                inputs.initial_state,
            )
            device_parameters = result.parameters
            mean_loss += Float32(result.loss)
            max_residual = max(max_residual, Float32(result.positive_residual))
        end
        mean_loss /= length(device_inputs)
        if epoch == 1 || epoch == epochs || epoch % max(1, epochs ÷ 4) == 0
            accuracy = reactant_linear_accuracy(device_parameters, input, labels)
            @printf(
                "epoch %3d | loss %.4f | accuracy %5.1f%% | residual %.2e\n",
                epoch, mean_loss, 100 * accuracy, max_residual,
            )
        end
    end

    accuracy = reactant_linear_accuracy(device_parameters, input, labels)
    return (; parameters=device_parameters, input, target, labels, accuracy,
            executable)
end

function parse_reactant_options(arguments)
    options = Dict(
        "backend" => "cpu",
        "epochs" => "12",
        "observations" => "256",
        "batch-size" => "32",
        "seed" => "7",
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
    options = parse_reactant_options(ARGS)
    result = train_reactant_ep(
        backend=options["backend"],
        epochs=parse(Int, options["epochs"]),
        observations=parse(Int, options["observations"]),
        batch_size=parse(Int, options["batch-size"]),
        seed=parse(Int, options["seed"]),
    )
    @printf "final accuracy: %5.1f%%\n" (100 * result.accuracy)
end
