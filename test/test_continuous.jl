@testset "continuous EP" begin
    problem, _, _ = quadratic_problem()
    β = 0.02
    solver = Relaxation(dt=0.5, maxiters=200, abstol=1e-11, reltol=0.0)
    algorithm = ContinuousEP(
        β,
        solver;
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
        parameter_abstol=1e-14,
        parameter_reltol=0.0,
    )

    η = 1e-8
    update_count = Ref(0)
    updater = function (parameters, gradient, iteration)
        update_count[] += 1
        return parameters .- η .* gradient
    end
    parameters, stats = continuous_ep(problem, algorithm; update_parameters=updater)
    @test update_count[] == stats.iterations
    @test stats.iterations > 1
    @test stats.free.converged
    @test stats.state_converged
    @test stats.parameters_stable
    @test stats.converged
    @test length(stats.residual_history) == stats.iterations
    @test length(stats.parameter_motion_history) == stats.iterations
    @test stats.parameter_displacement ≈ norm(parameters .- problem.parameters)
    @test solver_details(stats.nudged).residual_history === stats.residual_history

    standard_algorithm = EPAlgorithm(
        OneSidedEP(β),
        solver;
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    standard_gradient = ep_gradient(problem, standard_algorithm)[1]
    accumulated_gradient = (problem.parameters .- parameters) ./ η
    @test accumulated_gradient ≈ standard_gradient rtol=2e-5 atol=2e-8

    nudging_calls = Int[]
    learning_calls = Int[]
    scheduled = ContinuousEP(
        β,
        Relaxation(dt=0.2, maxiters=3, abstol=0.0, reltol=0.0);
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
        nudging_schedule=(iteration, base) -> begin
            push!(nudging_calls, iteration)
            base / iteration
        end,
        learning_rate_schedule=iteration -> begin
            push!(learning_calls, iteration)
            inv(iteration)
        end,
        parameter_abstol=0.0,
        parameter_reltol=0.0,
    )
    _, scheduled_stats = continuous_ep(
        problem,
        scheduled;
        update_parameters=(parameters, gradient, iteration) ->
            parameters .- 1e-4 .* gradient,
    )
    @test scheduled_stats.iterations == 3
    @test nudging_calls == [1, 2, 3]
    @test learning_calls == [1, 2, 3]
    @test scheduled_stats.β == β / 3

    invalid_nudging = ContinuousEP(
        β,
        solver;
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
        nudging_schedule=(iteration, base) -> 0.0,
    )
    @test_throws ArgumentError continuous_ep(
        problem,
        invalid_nudging;
        update_parameters=(parameters, gradient, iteration) -> parameters,
    )
    @test_throws ArgumentError ContinuousEP(
        0.0,
        solver;
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )

    optimizer_state = Optimisers.setup(Optimisers.Descent(1e-4), problem.parameters)
    optimizer_state, optimized_parameters, optimizer_stats = continuous_train_step!(
        optimizer_state,
        problem,
        ContinuousEP(
            β,
            Relaxation(dt=0.5, maxiters=20, abstol=1e-8);
            state_ad=AutoForwardDiff(),
            parameter_ad=AutoForwardDiff(),
        ),
    )
    @test optimizer_state !== nothing
    @test optimized_parameters != problem.parameters
    @test optimizer_stats.iterations > 0

    optimizer_state = Optimisers.setup(Optimisers.Descent(1e-4), problem.parameters)
    _, optimized_parameters2, _ = continuous_train_step!(
        optimizer_state,
        problem.parameters,
        problem.model,
        (problem.input, problem.target),
        algorithm;
        model_state=problem.model_state,
    )
    @test optimized_parameters2 != problem.parameters

    network = ContinuousHopfield(
        (2, 2, 1);
        activation=identity,
        weight_initializer=(rng, rows, columns) -> fill(0.05, rows, columns),
    )
    hopfield_model, hopfield_parameters = EquilibriumPropagation.setup(
        Xoshiro(4), network,
    )
    hopfield_problem = EPProblem(
        hopfield_model,
        hopfield_parameters,
        NamedTuple(),
        [0.2, -0.1],
        [0.5],
    )
    hopfield_algorithm = ContinuousEP(
        0.05,
        Relaxation(dt=0.2, maxiters=5, abstol=1e-8);
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    hopfield_optimizer = Optimisers.setup(
        Optimisers.Descent(1e-3), hopfield_parameters,
    )
    _, updated_hopfield_parameters, hopfield_stats = continuous_train_step!(
        hopfield_optimizer,
        hopfield_problem,
        hopfield_algorithm,
    )
    @test size.(updated_hopfield_parameters.weights) == ((2, 2), (1, 2))
    @test updated_hopfield_parameters.weights != hopfield_parameters.weights
    @test hopfield_stats.iterations == 5
end
