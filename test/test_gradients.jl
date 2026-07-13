@testset "EP gradients and diagnostics" begin
    problem, drive, y = quadratic_problem()
    nstate, ninput = length(y), length(problem.input)
    exact_drive_gradient = drive .- y
    exact = [vec(exact_drive_gradient * problem.input'); exact_drive_gradient]

    symmetric = quadratic_algorithm(SymmetricEP(0.01))
    phases = solve_phases(problem, symmetric)
    estimate = ep_gradient(problem, phases, symmetric)
    @test estimate ≈ exact rtol=2e-4 atol=2e-8

    estimate2, stats = ep_gradient(problem, symmetric)
    @test estimate2 == estimate
    @test stats.converged
    @test stats.loss ≈ sum(abs2, drive .- y) / 2 atol=1e-10
    @test stats.gradient_norm ≈ norm(estimate)
    @test stats.negative_residual isa Real
    @test stats.phase_displacement.positive > 0
    @test stats.energy_free == phases.free.energy

    one_sided = quadratic_algorithm(OneSidedEP(1e-3))
    one_estimate, one_stats = ep_gradient(problem, one_sided)
    @test one_estimate ≈ exact rtol=2e-3 atol=2e-7
    @test one_stats.negative_residual === missing
    @test one_stats.negative_iterations === missing
    @test one_stats.energy_negative === missing

    # Symmetric EP's finite-β error and solve error should decrease independently.
    coarse = ep_gradient(problem, quadratic_algorithm(SymmetricEP(0.2)))[1]
    fine = ep_gradient(problem, quadratic_algorithm(SymmetricEP(0.05)))[1]
    @test norm(fine .- exact) < norm(coarse .- exact)

    loose_algorithm = EPAlgorithm(
        SymmetricEP(0.05),
        Relaxation(dt=0.5, maxiters=2, abstol=0.0);
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    loose = ep_gradient(problem, loose_algorithm)[1]
    tight = ep_gradient(problem, quadratic_algorithm(SymmetricEP(0.05)))[1]
    @test norm(tight .- exact) < norm(loose .- exact)
end
