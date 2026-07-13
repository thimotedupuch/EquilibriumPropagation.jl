@testset "optional SciML relaxation extensions" begin
    problem, drive, target = quadratic_problem()

    tsit = ODERelaxation(
        OrdinaryDiffEqTsit5.Tsit5();
        tspan=(0.0, 50.0),
        abstol=1e-9,
        reltol=1e-7,
    )
    tsit_solution = equilibrate(problem, FreePhase(), tsit; state_ad=AutoForwardDiff())
    @test tsit_solution.converged
    @test tsit_solution.state ≈ drive atol=2e-7
    @test tsit_solution.iterations > 0
    @test SciMLBase.successful_retcode(solver_details(tsit_solution).retcode)
    @test CommonSolve.solve(problem, tsit; state_ad=AutoForwardDiff()).converged

    short = ODERelaxation(
        OrdinaryDiffEqTsit5.Tsit5();
        tspan=(0.0, 1e-4),
        abstol=1e-12,
        reltol=0.0,
    )
    @test !equilibrate(problem, FreePhase(), short; state_ad=AutoForwardDiff()).converged

    rodas = ODERelaxation(
        OrdinaryDiffEqRosenbrock.Rodas5P();
        tspan=(0.0, 50.0),
        abstol=1e-9,
        reltol=1e-7,
    )
    rodas_solution = equilibrate(problem, FreePhase(), rodas; state_ad=AutoForwardDiff())
    @test rodas_solution.converged
    @test rodas_solution.state ≈ drive atol=2e-7

    steady = SteadyStateRelaxation(
        OrdinaryDiffEqRosenbrock.Rodas5P();
        tspan=50.0,
        abstol=1e-9,
        reltol=1e-7,
    )
    steady_solution = equilibrate(
        problem, FreePhase(), steady; state_ad=AutoForwardDiff(),
    )
    @test steady_solution.converged
    @test steady_solution.state ≈ drive atol=2e-7
    @test SciMLBase.successful_retcode(solver_details(steady_solution).retcode)

    root = RootRelaxation(
        NonlinearSolve.TrustRegion(); abstol=1e-10, reltol=1e-8,
    )
    root_solution = equilibrate(problem, FreePhase(), root; state_ad=AutoForwardDiff())
    @test root_solution.converged
    @test root_solution.state ≈ drive atol=2e-9
    @test root_solution.residual <= root.abstol + root.reltol * norm(drive)

    root_algorithm = EPAlgorithm(
        SymmetricEP(0.05),
        root;
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    phases = solve_phases(problem, root_algorithm)
    @test phases.free.converged
    @test phases.positive.state ≈ (drive .+ 0.05 .* target) ./ 1.05 atol=2e-8
    @test phases.negative.state ≈ (drive .- 0.05 .* target) ./ 0.95 atol=2e-8

    tree_model = EPModel(
        energy=(state, parameters, input, model_state) -> sum(abs2, state.x),
        cost=(state, target, parameters, model_state) -> sum(abs2, state.x),
        readout=(state, parameters, model_state) -> state.x,
        initial_state=(parameters, input, model_state) -> (x=zeros(2),),
    )
    tree_problem = EPProblem(tree_model, [1.0], NamedTuple(), [1.0], [1.0])
    @test_throws ArgumentError equilibrate(
        tree_problem, FreePhase(), tsit; state_ad=AutoForwardDiff(),
    )
    @test_throws ArgumentError equilibrate(
        tree_problem, FreePhase(), steady; state_ad=AutoForwardDiff(),
    )
    @test_throws ArgumentError equilibrate(
        tree_problem, FreePhase(), root; state_ad=AutoForwardDiff(),
    )
end
