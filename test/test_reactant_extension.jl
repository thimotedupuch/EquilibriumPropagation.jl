using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using Enzyme
using ForwardDiff
using Optimisers
using Reactant
using Test

reactant_test_energy(s, ps, x, st) = sum(abs2, s) / 2 - sum(s .* (ps * x))
reactant_test_cost(s, y, ps, st) = sum(abs2, s .- y) / (2 * size(y, 2))
reactant_test_readout(s, ps, st) = s
reactant_test_initial(ps, x, st) = zero(ps * x)

function reactant_tree_energy(s, ps, x, st)
    return sum(abs2, s.hidden) / 2 + sum(abs2, s.output) / 2 -
           sum(s.hidden .* (ps.input * x)) -
           sum(s.output .* (ps.readout * s.hidden))
end
reactant_tree_cost(s, y, ps, st) = sum(abs2, s.output .- y) / (2 * size(y, 2))
reactant_tree_readout(s, ps, st) = s.output
function reactant_tree_initial(ps, x, st)
    hidden = zero(ps.input * x)
    return (hidden=hidden, output=zero(ps.readout * hidden))
end

reactant_field(s, ps, x, st) = st.matrix * s + ps
reactant_field_cost(s, y, ps, st) = sum(abs2, s .- y) / 2
reactant_field_readout(s, ps, st) = s
reactant_field_initial(ps, x, st) = zero(ps)

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
    @test Int(result.free_iterations) == compiled_algorithm.free_steps
    @test Bool(result.converged) == false

    rk4_algorithm = ReactantEP(
        OneSidedEP(0.1f0); method=:rk4, dt=0.5f0, free_steps=8,
        nudged_steps=6, abstol=1f-5, reltol=1f-5,
    )
    rk4_result = compile_reactant(problem, rk4_algorithm; backend="cpu")()
    @test all(isfinite, Array(rk4_result.free_state))
    @test Float32(rk4_result.free_residual) < 1f-2
    @test Int(rk4_result.free_iterations) <= rk4_algorithm.free_steps

    newton_algorithm = ReactantEP(
        OneSidedEP(0.1f0); method=:newton, free_steps=4, nudged_steps=4,
        abstol=1f-6, reltol=1f-6, damping=1f-4, step_scale=1f0,
    )
    newton_result = compile_reactant(problem, newton_algorithm; backend="cpu")()
    expected_free = parameters * input
    @test Array(newton_result.free_state) ≈ expected_free rtol=2f-5 atol=2f-6
    @test Float32(newton_result.free_residual) < 1f-5
    @test Int(newton_result.free_iterations) < newton_algorithm.free_steps
    @test all(isfinite, Array(newton_result.gradient))

    tree_parameters = (
        input=Float32[0.2 -0.1; 0.3 0.4; -0.2 0.1],
        readout=Float32[0.1 -0.3 0.2; -0.2 0.2 0.4],
    )
    tree_model = EPModel(
        energy=reactant_tree_energy,
        cost=reactant_tree_cost,
        readout=reactant_tree_readout,
        initial_state=reactant_tree_initial,
    )
    tree_problem = EPProblem(
        tree_model,
        tree_parameters,
        NamedTuple(),
        EPBatch(input, target),
    )
    stopping_algorithm = ReactantEP(
        OneSidedEP(0.1f0);
        dt=0.2f0,
        free_steps=8,
        nudged_steps=6,
        abstol=100.0f0,
        reltol=0.0f0,
    )
    tree_result = compile_reactant(
        tree_problem, stopping_algorithm; backend="cpu",
    )()
    @test tree_result.free_state isa NamedTuple
    @test keys(tree_result.free_state) == (:hidden, :output)
    @test keys(tree_result.gradient) == keys(tree_parameters)
    @test Int(tree_result.free_iterations) == 0
    @test Int(tree_result.positive_iterations) == 0
    @test Bool(tree_result.converged)
    @test all(isfinite, Array(tree_result.gradient.input))
    @test all(isfinite, Array(tree_result.gradient.readout))

    field_model = DynamicalModel(
        dynamics=reactant_field,
        cost=reactant_field_cost,
        readout=reactant_field_readout,
        initial_state=reactant_field_initial,
    )
    field_problem = EPProblem(
        field_model,
        Float32[0.3, -0.2],
        (matrix=Float32[-2 1; 0 -1],),
        nothing,
        Float32[-0.4, 0.7],
    )
    field_solver = Relaxation(
        dt=0.1f0, maxiters=20, abstol=0.0f0, reltol=0.0f0,
    )
    field_backend = AutoForwardDiff()
    asym = AsymEP(
        1f-3, field_solver;
        state_ad=field_backend, parameter_ad=field_backend,
    )
    compiled_asym = compile_reactant(
        field_problem, asym, Optimisers.Adam(1f-2); backend="cpu",
    )()
    reference_asym = ep_gradient(field_problem, asym)[1]
    @test all(isfinite, Array(compiled_asym.gradient))
    @test Array(compiled_asym.gradient) ≈ reference_asym rtol=3f-4 atol=3f-5
    @test Array(compiled_asym.parameters) != field_problem.parameters
    @test Int(compiled_asym.free_iterations) == field_solver.maxiters

    dyadic = DyadicEP(
        1f-3, field_solver;
        state_ad=field_backend, parameter_ad=field_backend,
    )
    compiled_dyadic = compile_reactant(
        field_problem, dyadic; backend="cpu",
    )()
    reference_dyadic = ep_gradient(field_problem, dyadic)[1]
    @test all(isfinite, Array(compiled_dyadic.gradient))
    @test Array(compiled_dyadic.gradient) ≈ reference_dyadic rtol=3f-4 atol=3f-5
    @test size(Array(compiled_dyadic.difference_state)) == size(field_problem.parameters)

    continuous = ContinuousEP(
        0.1f0,
        Relaxation(dt=0.2f0, maxiters=2, abstol=0.0f0);
        state_ad=field_backend,
        parameter_ad=field_backend,
    )
    compiled_continuous = compile_reactant(
        problem, continuous, Optimisers.Descent(1f-2); backend="cpu",
    )()
    reference_optimizer = Optimisers.setup(Optimisers.Descent(1f-2), parameters)
    _, reference_continuous_parameters, _ = continuous_train_step!(
        reference_optimizer, problem, continuous,
    )
    @test Int(compiled_continuous.iterations) == 2
    @test Array(compiled_continuous.parameters) != parameters
    @test Array(compiled_continuous.parameters) ≈
          reference_continuous_parameters rtol=3f-4 atol=3f-5

    holomorphic = ReactantEP(
        HolomorphicEP(0.1f0; points=4); free_steps=2, nudged_steps=2,
    )
    @test_throws ArgumentError compile_reactant(
        problem, holomorphic; backend="cpu",
    )
end
