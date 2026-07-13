@testset "feedforward AsymEP MLP example" begin
    include(joinpath(@__DIR__, "..", "examples", "asymep_mlp.jl"))
    rng = Xoshiro(19)
    inputs, targets, labels = mlp_spiral_dataset(rng, 8)
    model, parameters, model_state = asymep_mlp(3; rng)
    problem = EPProblem(model, parameters, model_state, inputs, targets)
    solver = asymep_mlp_relaxation(maxiters=20, abstol=1f-4)
    backend = AutoForwardDiff()
    gradient, stats = ep_gradient(
        problem,
        AsymEP(0.1f0, solver; state_ad=backend, parameter_ad=backend),
    )
    @test size(gradient.input_weight) == (3, 2)
    @test size(gradient.output_weight) == (2, 3)
    @test all(isfinite, Iterators.flatten(vec.(values(gradient))))
    @test isfinite(stats.loss)
    accuracy, solution = asymep_mlp_accuracy(
        model, parameters, model_state, inputs, labels,
    )
    @test 0 <= accuracy <= 1
    @test solution.converged
end
