@testset "continuous Hopfield builder" begin
    rng = Random.MersenneTwister(12)
    specification = ContinuousHopfield((3, 4, 2))
    model, parameters = EquilibriumPropagation.setup(rng, specification)

    @test size.(parameters.weights) == ((4, 3), (2, 4))
    @test size.(parameters.biases) == ((4,), (2,))
    @test parameters.recurrent == ()

    input = randn(rng, Float32, 3, 5)
    target = randn(rng, Float32, 2, 5)
    state = initial_state(model, parameters, NamedTuple(), input)
    @test size(state) == (6, 5)
    @test iszero(state)
    @test energy(model, state, parameters, NamedTuple(), input) isa Number
    @test cost(model, state, parameters, NamedTuple(), target) ≈
          sum(abs2, target) / (2 * size(target, 2))
    @test size(readout(model, state, parameters, NamedTuple())) == (2, 5)

    packed = pack_parameters(specification, parameters)
    rebuilt = unpack_parameters(specification, packed)
    @test rebuilt == parameters

    vector_input = randn(rng, Float32, 3)
    vector_state = initial_state(model, parameters, NamedTuple(), vector_input)
    @test size(vector_state) == (6,)
    @test size(readout(model, vector_state, parameters, NamedTuple())) == (2,)

    @test_throws DimensionMismatch initial_state(
        model, parameters, NamedTuple(), zeros(Float32, 4, 1),
    )
    @test_throws DimensionMismatch cost(
        model, state, parameters, NamedTuple(), zeros(Float32, 3, 5),
    )
    @test_throws DimensionMismatch EPProblem(
        model, parameters, NamedTuple(), input, zeros(Float32, 3, 5),
    )
    @test_throws DimensionMismatch EPProblem(
        model, parameters, NamedTuple(), input, zeros(Float32, 2, 4),
    )

    recurrent_specification = ContinuousHopfield(
        (2, 3, 1); bias=false, recurrent=true,
    )
    recurrent_model, recurrent_parameters = EquilibriumPropagation.setup(
        rng, recurrent_specification,
    )
    @test recurrent_parameters.biases == ()
    @test all(issymmetric, recurrent_parameters.recurrent)
    @test all(matrix -> iszero(diag(matrix)), recurrent_parameters.recurrent)

    # Only the symmetric, zero-diagonal part can affect the scalar energy.
    changed_parameters = merge(recurrent_parameters, (
        recurrent=map(recurrent_parameters.recurrent) do matrix
            matrix + Float32[10 2 -3; -2 20 4; 3 -4 30][1:size(matrix, 1), 1:size(matrix, 2)]
        end,
    ))
    recurrent_input = randn(rng, Float32, 2, 2)
    recurrent_state = randn(rng, Float32, 4, 2)
    @test energy(
        recurrent_model, recurrent_state, recurrent_parameters, NamedTuple(), recurrent_input,
    ) ≈ energy(
        recurrent_model, recurrent_state, changed_parameters, NamedTuple(), recurrent_input,
    )

    custom_state = (parameters, input, model_state) ->
        fill(eltype(input)(0.25), 3, size(input, 2))
    custom_specification = ContinuousHopfield(
        (2, 3);
        activation=identity,
        potential=x -> abs(x),
        output_cost=(output, target) -> sum(abs, output .- target) / size(target, 2),
        state_initializer=custom_state,
        weight_initializer=(rng, rows, columns) -> fill(0.5, rows, columns),
        bias_initializer=(rng, width) -> ones(width),
    )
    custom_model, custom_parameters = EquilibriumPropagation.setup(rng, custom_specification)
    custom_input = zeros(2, 4)
    @test initial_state(custom_model, custom_parameters, NamedTuple(), custom_input) ==
          fill(0.25, 3, 4)
    @test custom_parameters.weights[1] == fill(0.5, 3, 2)

    ep_specification = ContinuousHopfield(
        (2, 3, 1);
        activation=identity,
        weight_initializer=(rng, rows, columns) -> fill(0.05f0, rows, columns),
    )
    ep_model, ep_parameters = EquilibriumPropagation.setup(rng, ep_specification)
    ep_problem = EPProblem(
        ep_model,
        ep_parameters,
        NamedTuple(),
        Float32[0.2, -0.1],
        Float32[0.5],
    )
    ep_algorithm = EPAlgorithm(
        SymmetricEP(0.05f0),
        Relaxation(dt=0.2f0, maxiters=500, abstol=1f-5);
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    ep_gradients, ep_stats = ep_gradient(ep_problem, ep_algorithm)
    @test size.(ep_gradients.weights) == ((3, 2), (1, 3))
    @test size.(ep_gradients.biases) == ((3,), (1,))
    @test ep_gradients.recurrent == ()
    @test all(isfinite, Iterators.flatten(vec.(ep_gradients.weights)))
    @test ep_stats.converged

    optimizer_state = Optimisers.setup(Optimisers.Descent(0.01f0), ep_parameters)
    optimizer_state, updated_parameters = apply_gradient!(
        optimizer_state, ep_parameters, ep_gradients,
    )
    @test updated_parameters.weights != ep_parameters.weights

    adjacency = Bool[
        0 0 1 0 0
        0 0 0 1 0
        1 0 0 1 0
        0 1 1 0 1
        0 0 0 1 0
    ]
    graph_specification = AdjacencyHopfield(
        adjacency;
        input_size=2,
        output_size=1,
        activation=identity,
        weight_initializer=(rng, rows, columns) -> ones(Float32, rows, columns),
        bias_initializer=(rng, width) -> zeros(Float32, width),
    )
    graph_model, graph_parameters = EquilibriumPropagation.setup(
        rng, graph_specification,
    )
    @test graph_parameters.coupling == Float32.(adjacency)
    @test graph_parameters.bias == zeros(Float32, 3)
    @test all(iszero, graph_parameters.coupling[.!adjacency])

    graph_input = Float32[2, 3]
    graph_state = Float32[5, 7, 11]
    graph_target = Float32[13]
    @test initial_state(
        graph_model, graph_parameters, NamedTuple(), graph_input,
    ) == zeros(Float32, 3)
    @test readout(graph_model, graph_state, graph_parameters, NamedTuple()) == Float32[11]
    all_rates = vcat(graph_input, graph_state)
    expected_energy = sum(abs2, graph_state) / 2 -
                      dot(all_rates, Float32.(adjacency) * all_rates) / 2
    @test energy(
        graph_model, graph_state, graph_parameters, NamedTuple(), graph_input,
    ) ≈ expected_energy
    @test cost(
        graph_model, graph_state, graph_parameters, NamedTuple(), graph_target,
    ) ≈ 2

    graph_packed = pack_parameters(graph_specification, graph_parameters)
    @test unpack_parameters(graph_specification, graph_packed) == graph_parameters
    graph_problem = EPProblem(
        graph_model, graph_parameters, NamedTuple(), graph_input, graph_target,
    )
    graph_algorithm = EPAlgorithm(
        OneSidedEP(0.05f0), Relaxation(dt=0.02f0, maxiters=10, abstol=0f0);
        state_ad=AutoForwardDiff(), parameter_ad=AutoForwardDiff(),
    )
    graph_gradient, _ = ep_gradient(graph_problem, graph_algorithm)
    @test all(iszero, graph_gradient.coupling[.!adjacency])

    @test_throws ArgumentError ContinuousHopfield((3,))
    @test_throws ArgumentError ContinuousHopfield((3, 0, 2))
    @test_throws ArgumentError ContinuousHopfield((3, 2); input_clamp=:soft)
    @test_throws DimensionMismatch AdjacencyHopfield(
        zeros(Bool, 2, 3); input_size=1, output_size=1,
    )
    @test_throws ArgumentError AdjacencyHopfield(
        [0 2; 2 0]; input_size=1, output_size=1,
    )
    @test_throws ArgumentError AdjacencyHopfield(
        [0 1; 0 0]; input_size=1, output_size=1,
    )
    @test_throws ArgumentError AdjacencyHopfield(
        [1 0; 0 0]; input_size=1, output_size=1,
    )
    @test_throws DimensionMismatch EquilibriumPropagation.setup(
        rng,
        ContinuousHopfield(
            (3, 2);
            weight_initializer=(rng, rows, columns) -> zeros(rows + 1, columns),
        ),
    )
end
