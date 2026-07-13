@testset "continuous Hopfield spiral example" begin
    include(joinpath(@__DIR__, "..", "examples", "continuous_hopfield_spiral.jl"))
    rng = Xoshiro(11)
    inputs, targets, labels = spiral_dataset(rng, 8)
    @test size(inputs) == (2, 8)
    @test size(targets) == (2, 8)
    @test sort(unique(labels)) == [1, 2]

    layout = HopfieldLayout(2, 4, 2)
    model = continuous_hopfield_model(layout)
    parameters = initialize_parameters(rng, layout)
    @test eltype(parameters) == Float32

    problem = EPProblem(model, parameters, NamedTuple(), inputs, targets)
    algorithm = EPAlgorithm(
        SymmetricEP(0.1f0),
        relaxation_for(8; maxiters=5, abstol=0f0);
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    gradients, stats = ep_gradient(problem, algorithm)
    @test size(gradients) == size(parameters)
    @test all(isfinite, gradients)
    @test isfinite(stats.loss)
end
