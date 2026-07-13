function linear_dynamical_problem(parameters=[0.3, -0.2])
    matrix = [-2.0 1.0; 0.0 -1.0]
    target = [-0.4, 0.7]
    model = DynamicalModel(
        dynamics=(state, ps, input, model_state) -> matrix * state + ps,
        cost=(state, y, ps, model_state) ->
            sum(abs2, state .- y) / 2 + 0.1 * sum(abs2, ps),
        readout=(state, ps, model_state) -> state,
        initial_state=(ps, input, model_state) -> zeros(2),
    )
    return EPProblem(model, parameters, NamedTuple(), nothing, target), matrix
end

@testset "Non-conservative equilibrium propagation" begin
    problem, matrix = linear_dynamical_problem()
    solver = Relaxation(dt=0.1, maxiters=1000, abstol=1e-11)
    backend = AutoForwardDiff()
    asym = AsymEP(1e-4, solver; state_ad=backend, parameter_ad=backend)
    dyadic = DyadicEP(1e-4, solver; state_ad=backend, parameter_ad=backend)

    exact = ForwardDiff.gradient(problem.parameters) do parameters
        equilibrium = -(matrix \ parameters)
        cost(problem.model, equilibrium, parameters,
             problem.model_state, problem.target)
    end
    asym_gradient, asym_stats = ep_gradient(problem, asym)
    dyadic_gradient, dyadic_stats = ep_gradient(problem, dyadic)

    @test asym_gradient ≈ exact rtol=2e-5 atol=2e-8
    @test dyadic_gradient ≈ exact rtol=2e-7 atol=2e-7
    @test asym_stats.method === :AsymEP
    @test dyadic_stats.method === :DyadicEP
    @test asym_stats.converged
    @test dyadic_stats.converged
    @test asym_stats.jacobian_asymmetry > 0
    @test dyadic_stats.jacobian_asymmetry ≈ asym_stats.jacobian_asymmetry
    @test dyadic_stats.positive.state.midpoint ≈ dyadic_stats.free.state atol=1e-10
    @test predict(asym_stats.free, problem) ≈ -(matrix \ problem.parameters) atol=1e-9

    @test_throws ArgumentError AsymEP(0.0, solver; state_ad=backend, parameter_ad=backend)
    @test_throws ArgumentError DyadicEP(0.0, solver; state_ad=backend, parameter_ad=backend)

    tree_parameters = (drive=[0.3, -0.2],)
    tree_model = DynamicalModel(
        dynamics=(state, ps, input, model_state) -> matrix * state + ps.drive,
        cost=(state, y, ps, model_state) -> sum(abs2, state .- y) / 2,
        readout=(state, ps, model_state) -> state,
        initial_state=(ps, input, model_state) -> zeros(2),
    )
    tree_problem = EPProblem(tree_model, tree_parameters, NamedTuple(), nothing,
                             problem.target)
    tree_gradient, tree_stats = ep_gradient(tree_problem, dyadic)
    @test tree_gradient.drive ≈ -(transpose(matrix) \ (tree_stats.free.state .-
                                                         problem.target)) atol=2e-7

    optimizer_state = Optimisers.setup(Optimisers.Descent(0.01), problem.parameters)
    optimizer_state, updated, optimizer_stats = train_step!(optimizer_state, problem, asym)
    @test updated ≈ problem.parameters .- 0.01 .* asym_gradient
    @test optimizer_stats.method === :AsymEP
end

@testset "Conservative reduction" begin
    matrix = [-2.0 0.4; 0.4 -1.5]
    parameters = [0.2, -0.3]
    target = [-0.1, 0.6]
    dynamical = DynamicalModel(
        dynamics=(state, ps, input, model_state) -> matrix * state + ps,
        cost=(state, y, ps, model_state) -> sum(abs2, state .- y) / 2,
        readout=(state, ps, model_state) -> state,
        initial_state=(ps, input, model_state) -> zeros(2),
    )
    conservative = EPModel(
        energy=(state, ps, input, model_state) ->
            -dot(state, matrix * state) / 2 - dot(ps, state),
        cost=(state, y, ps, model_state) -> sum(abs2, state .- y) / 2,
        readout=(state, ps, model_state) -> state,
        initial_state=(ps, input, model_state) -> zeros(2),
    )
    dynamic_problem = EPProblem(dynamical, parameters, NamedTuple(), nothing, target)
    energy_problem = EPProblem(conservative, parameters, NamedTuple(), nothing, target)
    solver = Relaxation(dt=0.1, maxiters=1000, abstol=1e-11)
    backend = AutoForwardDiff()
    asym_gradient, stats = ep_gradient(
        dynamic_problem,
        AsymEP(1e-4, solver; state_ad=backend, parameter_ad=backend),
    )
    ep_result, _ = ep_gradient(
        energy_problem,
        EPAlgorithm(SymmetricEP(1e-4), solver;
                    state_ad=backend, parameter_ad=backend),
    )
    @test stats.jacobian_asymmetry ≈ 0 atol=1e-14
    @test asym_gradient ≈ ep_result rtol=2e-5 atol=2e-8
end
