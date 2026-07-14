include("setup.jl")

@testset "model interface" begin
    problem, drive, _ = quadratic_problem()
    s0 = initial_state(
        problem.model,
        problem.parameters,
        problem.model_state,
        problem.input,
    )
    @test s0 == zeros(3)
    @test energy(problem.model, drive, problem.parameters, problem.model_state, problem.input) isa Number
    @test readout(problem.model, drive, problem.parameters, problem.model_state) === drive
    @test augmented_energy(problem, drive, 0.0) ==
          energy(problem.model, drive, problem.parameters, problem.model_state, problem.input)

    alias_model = EPModel(
        energy=(s, ps, x, st) -> sum(s),
        cost=(s, y, ps, st) -> sum(s),
        readout=(s, ps, st) -> s,
        initialstate=(ps, x, st) -> zero(x),
    )
    @test initial_state(alias_model, nothing, nothing, [1.0]) == [0.0]

    batch = EPBatch(randn(3, 4), randn(2, 4))
    @test batch_size(batch) == 4
    @test batch_size(EPBatch(randn(3), randn(2))) == 1
    batched_problem = EPProblem(problem.model, problem.parameters,
                                problem.model_state, batch)
    @test batched_problem.input === batch.input
    @test batched_problem.target === batch.target
    @test_throws DimensionMismatch EPBatch(randn(3, 2, 1), randn(2, 2, 1))
    @test_throws DimensionMismatch EPBatch(randn(3, 2), randn(2))
    @test_throws DimensionMismatch EPBatch(randn(3, 2), randn(2, 3))

    @test_throws ArgumentError OneSidedEP(0.0)
    @test_throws ArgumentError SymmetricEP(0.0)
    @test_throws ArgumentError Relaxation(dt=0.0)
    @test ReactantEP(SymmetricEP(0.1); dt=0.2, free_steps=3,
                     nudged_steps=4, learning_rate=0.01).free_steps == 3
    @test_throws ArgumentError ReactantEP(OneSidedEP(0.1); dt=0.0)
    @test_throws ArgumentError ReactantEP(OneSidedEP(0.1); free_steps=-1)
    @test_throws ArgumentError ReactantEP(OneSidedEP(0.1); nudged_steps=-1)
    @test_throws ArgumentError ReactantEP(OneSidedEP(0.1); abstol=-1)
    @test_throws ArgumentError ReactantEP(OneSidedEP(0.1); reltol=-1)
    @test_throws ArgumentError ReactantEP(OneSidedEP(0.1); learning_rate=-0.1)
end
