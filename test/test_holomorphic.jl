@testset "Holomorphic EP" begin
    parameters = [0.3, -0.2]
    target = [-0.4, 0.7]
    model = EPModel(
        energy=(state, ps, input, model_state) ->
            sum(state .^ 2) / 2 - sum(ps .* state),
        cost=(state, y, ps, model_state) -> sum((state .- y) .^ 2) / 2,
        readout=(state, ps, model_state) -> state,
        initial_state=(ps, input, model_state) -> zeros(length(ps)),
    )
    problem = EPProblem(model, parameters, NamedTuple(), nothing, target)
    solver = Relaxation(dt=0.5, maxiters=300, abstol=1e-12)
    backend = AutoForwardDiff()
    protocol = HolomorphicEP(0.2; points=16)
    algorithm = EPAlgorithm(protocol, solver; state_ad=backend, parameter_ad=backend)
    gradient, stats = ep_gradient(problem, algorithm)

    @test gradient ≈ parameters .- target atol=2e-8
    @test stats.converged
    @test length(stats.phases.nudges) == 16
    @test length(stats.phases.solutions) == 16
    @test eltype(stats.phases.solutions[2].state) <: Complex
    @test stats.imaginary_leakage < 1e-8
    @test all(solution -> solution.phase isa NudgedPhase, stats.phases.solutions)
    @test predict(stats.phases.free, problem) ≈ parameters atol=1e-11

    two_point = EPAlgorithm(
        HolomorphicEP(1e-3; points=2), solver;
        state_ad=backend, parameter_ad=backend,
    )
    symmetric = EPAlgorithm(
        SymmetricEP(1e-3), solver;
        state_ad=backend, parameter_ad=backend,
    )
    holomorphic_gradient, _ = ep_gradient(problem, two_point)
    symmetric_gradient, _ = ep_gradient(problem, symmetric)
    @test holomorphic_gradient ≈ symmetric_gradient atol=2e-9

    tree_parameters = (drive=[0.3, -0.2], scale=0.1)
    tree_model = EPModel(
        energy=(state, ps, input, model_state) ->
            sum(state .^ 2) / 2 - sum(ps.drive .* state),
        cost=(state, y, ps, model_state) ->
            sum((state .- y) .^ 2) / 2 + ps.scale * sum(state),
        readout=(state, ps, model_state) -> state,
        initial_state=(ps, input, model_state) -> zeros(length(ps.drive)),
    )
    tree_problem = EPProblem(tree_model, tree_parameters, NamedTuple(), nothing, target)
    tree_gradient, tree_stats = ep_gradient(tree_problem, algorithm)
    exact_tree = (
        drive=tree_parameters.drive .- target .+ tree_parameters.scale,
        scale=sum(tree_parameters.drive),
    )
    @test tree_gradient.drive ≈ exact_tree.drive atol=2e-8
    @test tree_gradient.scale ≈ exact_tree.scale atol=2e-8
    @test tree_stats.converged

    @test_throws ArgumentError HolomorphicEP(0.0)
    @test_throws ArgumentError HolomorphicEP(0.1; points=1)
end
