using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using Enzyme
using ForwardDiff
using Reactant
using Test

reactant_test_energy(s, ps, x, st) = sum(abs2, s) / 2 - sum(s .* (ps * x))
reactant_test_cost(s, y, ps, st) = sum(abs2, s .- y) / (2 * size(y, 2))
reactant_test_readout(s, ps, st) = s
reactant_test_initial(ps, x, st) = zero(ps * x)

@testset "Reactant extension" begin
    Reactant.set_default_backend("cpu")
    parameters = Float32[0.25 -0.15; -0.1 0.3]
    input = Float32[0.8 -0.4 0.2; -0.3 0.5 0.9]
    target = Float32[0.6 -0.5 0.1; -0.2 0.4 0.7]
    model = EPModel(
        energy=reactant_test_energy,
        cost=reactant_test_cost,
        readout=reactant_test_readout,
        initial_state=reactant_test_initial,
    )
    problem = EPProblem(model, parameters, NamedTuple(), EPBatch(input, target))
    protocol = SymmetricEP(0.1f0)
    compiled_algorithm = ReactantEP(
        protocol;
        dt=0.25f0,
        free_steps=6,
        nudged_steps=5,
        learning_rate=0.02f0,
    )

    executable = compile_reactant(problem, compiled_algorithm; backend="cpu")
    result = executable()

    reference_algorithm = EPAlgorithm(
        protocol,
        Relaxation(dt=0.25f0, maxiters=5, abstol=0.0f0);
        free_solver=Relaxation(dt=0.25f0, maxiters=6, abstol=0.0f0),
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
    reference_gradient, reference_stats = ep_gradient(problem, reference_algorithm)
    reference_phases = solve_phases(problem, reference_algorithm)

    @test Array(result.gradient) ≈ reference_gradient rtol=2f-4 atol=2f-5
    @test Array(result.free_state) ≈ reference_phases.free.state rtol=2f-5 atol=2f-6
    @test Array(result.positive_state) ≈ reference_phases.positive.state rtol=2f-5 atol=2f-6
    @test Array(result.negative_state) ≈ reference_phases.negative.state rtol=2f-5 atol=2f-6
    @test Array(result.parameters) ≈
          parameters .- 0.02f0 .* reference_gradient rtol=2f-4 atol=2f-5
    @test Float32(result.loss) ≈ reference_stats.loss rtol=2f-5

    second_problem = EPProblem(
        model,
        parameters,
        NamedTuple(),
        EPBatch(input .+ 0.1f0, target),
    )
    second_inputs = reactant_inputs(second_problem; backend="cpu")
    second_result = executable(second_inputs)
    @test size(Array(second_result.gradient)) == size(parameters)
    @test all(isfinite, Array(second_result.gradient))
    reused_result = executable(
        result.parameters,
        second_inputs.model_state,
        second_inputs.batch,
        second_inputs.initial_state,
    )
    @test all(isfinite, Array(reused_result.parameters))
end
