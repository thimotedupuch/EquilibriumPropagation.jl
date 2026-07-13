@testset "hidden-unit MNIST Hopfield example" begin
    include(joinpath(@__DIR__, "..", "examples", "mnist_classification.jl"))
    rng = Xoshiro(23)
    network = mnist_classifier(4; hidden=3)
    @test network.layer_sizes == (4, 3, 10)
    model, parameters = EquilibriumPropagation.setup(rng, network)
    @test size.(parameters.weights) == ((3, 4), (10, 3))

    inputs = randn(rng, Float32, 4, 2)
    targets = -ones(Float32, 10, 2)
    targets[1, 1] = 1f0
    targets[2, 2] = 1f0
    problem = EPProblem(model, parameters, NamedTuple(), inputs, targets)
    backend = AutoForwardDiff()
    gradient, stats = ep_gradient(
        problem,
        EPAlgorithm(
            SymmetricEP(0.1f0),
            mnist_relaxation(2; maxiters=5, abstol=0f0);
            state_ad=backend,
            parameter_ad=backend,
        ),
    )
    @test size.(gradient.weights) == ((3, 4), (10, 3))
    @test isfinite(stats.loss)
    @test parse_mnist_options(["--hidden=17"])["hidden"] == 17
end
