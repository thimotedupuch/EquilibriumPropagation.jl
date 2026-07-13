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
    @test_throws ArgumentError OneSidedEP(0.0)
    @test_throws ArgumentError SymmetricEP(0.0)
    @test_throws ArgumentError Relaxation(dt=0.0)
end
