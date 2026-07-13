@testset "relaxation and phases" begin
    problem, drive, y = quadratic_problem()
    solver = Relaxation(dt=0.5, maxiters=200, abstol=1e-11)
    free = equilibrate(problem, FreePhase(), solver; state_ad=AutoForwardDiff())
    @test free.converged
    @test free.state ≈ drive atol=2e-11
    @test free.residual <= solver.abstol
    @test predict(free, problem) ≈ drive

    via_common_solve = CommonSolve.solve(problem, solver; state_ad=AutoForwardDiff())
    @test via_common_solve.state ≈ free.state

    β = 0.05
    algorithm = quadratic_algorithm(SymmetricEP(β))
    phases = solve_phases(problem, algorithm)
    @test phases.free.converged
    @test phases.positive.converged
    @test phases.negative.converged
    @test phases.positive.state ≈ (drive .+ β .* y) ./ (1 + β) atol=2e-11
    @test phases.negative.state ≈ (drive .- β .* y) ./ (1 - β) atol=2e-11

    one_sided = solve_phases(problem, quadratic_algorithm(OneSidedEP(β)))
    @test one_sided.negative === nothing

    phase_specific = EPAlgorithm(
        SymmetricEP(β),
        solver;
        positive_solver=Relaxation(dt=0.5, maxiters=1, abstol=0.0),
        negative_solver=Relaxation(dt=0.5, maxiters=2, abstol=0.0),
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    phase_specific_solutions = solve_phases(problem, phase_specific)
    @test phase_specific_solutions.positive.iterations == 1
    @test phase_specific_solutions.negative.iterations == 2

    unavailable = quadratic_algorithm(SymmetricEP(β))
    unavailable = EPAlgorithm(
        unavailable.protocol,
        unavailable.solver;
        state_ad=NoAutoDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    @test_throws ArgumentError solve_phases(problem, unavailable)
end
