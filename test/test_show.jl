@testset "REPL display" begin
    problem, _, _ = quadratic_problem()
    algorithm = quadratic_algorithm(SymmetricEP(0.05))
    phases = solve_phases(problem, algorithm)
    _, stats = ep_gradient(problem, algorithm)

    @test repr(FreePhase()) == "FreePhase()"
    @test repr(NudgedPhase(0.5)) == "NudgedPhase(0.5)"
    @test repr(OneSidedEP(0.1)) == "OneSidedEP(0.1)"
    @test repr(SymmetricEP(0.1)) == "SymmetricEP(0.1)"
    @test repr(ReactantEP(SymmetricEP(0.1); dt=0.2, free_steps=3,
                         nudged_steps=4, learning_rate=0.01)) ==
          "ReactantEP(SymmetricEP(0.1); dt=0.2, free_steps=3, nudged_steps=4, learning_rate=0.01)"
    @test repr(Relaxation()) ==
          "Relaxation(dt=0.1, maxiters=100, abstol=1.0e-6, reltol=0.0)"
    @test startswith(repr(problem), "EPProblem(model=EPModel(")
    @test !occursin("EPProblem{", repr(problem))

    compact_stats = repr(stats)
    @test startswith(compact_stats, "EPStats(loss=")
    @test !occursin("EPStats{", compact_stats)

    stats_display = repr(MIME"text/plain"(), stats)
    @test startswith(stats_display, "EPStats\n  converged: true")
    @test occursin("\n  gradient norm:", stats_display)
    @test occursin("\n  phases:\n    free: residual=", stats_display)
    @test occursin("\n    negative: residual=", stats_display)
    @test !occursin("EPStats{", stats_display)

    solution_display = repr(MIME"text/plain"(), phases.free)
    @test startswith(solution_display, "EquilibriumSolution\n")
    @test occursin("\n  state: 3-element Vector{Float64}", solution_display)

    phases_display = repr(MIME"text/plain"(), phases)
    @test startswith(phases_display, "EPPhases\n  free: converged")
    @test occursin("\n  negative: converged", phases_display)

    one_sided = solve_phases(problem, quadratic_algorithm(OneSidedEP(0.05)))
    @test endswith(repr(MIME"text/plain"(), one_sided), "negative: not run")
end
