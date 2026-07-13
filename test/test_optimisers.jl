@testset "Optimisers extension" begin
    problem, _, _ = quadratic_problem()
    algorithm = quadratic_algorithm(SymmetricEP(0.01))
    optimizer_state = Optimisers.setup(Optimisers.Descent(0.1), problem.parameters)

    gradients, _ = ep_gradient(problem, algorithm)
    new_state, new_parameters, stats = train_step!(optimizer_state, problem, algorithm)
    @test new_parameters ≈ problem.parameters .- 0.1 .* gradients
    @test new_state !== nothing
    @test stats.converged

    state2 = Optimisers.setup(Optimisers.Descent(0.1), problem.parameters)
    _, parameters2, stats2 = train_step!(
        state2,
        problem.parameters,
        problem.model,
        (problem.input, problem.target),
        algorithm;
        model_state=problem.model_state,
    )
    @test parameters2 ≈ new_parameters
    @test stats2.loss ≈ stats.loss
end
