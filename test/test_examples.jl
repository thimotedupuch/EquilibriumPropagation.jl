@testset "continuous Hopfield spiral example" begin
    include(joinpath(@__DIR__, "..", "examples", "continuous_hopfield_spiral.jl"))
    rng = Xoshiro(11)
    inputs, targets, labels = spiral_dataset(rng, 8)
    @test size(inputs) == (2, 8)
    @test size(targets) == (2, 8)
    @test sort(unique(labels)) == [1, 2]

    network = spiral_network(4)
    model, parameters = EquilibriumPropagation.setup(rng, network)
    @test all(parameter -> eltype(parameter) == Float32, parameters.weights)

    problem = EPProblem(model, parameters, NamedTuple(), inputs, targets)
    algorithm = EPAlgorithm(
        SymmetricEP(0.1f0),
        relaxation_for(8; maxiters=5, abstol=0f0);
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    gradients, stats = ep_gradient(problem, algorithm)
    @test size.(gradients.weights) == ((4, 2), (2, 4))
    @test all(isfinite, Iterators.flatten(vec.(gradients.weights)))
    @test isfinite(stats.loss)
end
