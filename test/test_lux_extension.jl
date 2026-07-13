struct ChangingLuxLayer <: Lux.AbstractLuxLayer end

Lux.initialparameters(rng::Random.AbstractRNG, ::ChangingLuxLayer) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::ChangingLuxLayer) = (counter=0,)
(::ChangingLuxLayer)(input, parameters, state) = (input, (counter=state.counter + 1,))

@testset "Lux extension" begin
    rng = Xoshiro(42)
    layer = Lux.Dense(2 => 2, identity)
    parameters, model_state = lux_setup(rng, layer)
    matrix = [-2.0 1.0; 0.0 -1.0]
    bias = [0.3, -0.2]
    parameters = merge(parameters, (weight=matrix, bias=bias))
    target = [-0.4, 0.7]
    model = lux_dynamical_model(
        layer;
        cost=(state, y, ps, st) -> sum(abs2, state .- y) / 2,
        readout=(state, ps, st) -> state,
        initial_state=(ps, input, st) -> zeros(2),
    )
    problem = EPProblem(model, parameters, model_state, nothing, target)
    solver = Relaxation(dt=0.1, maxiters=1000, abstol=1e-11)
    backend = AutoForwardDiff()
    gradient, stats = ep_gradient(
        problem,
        DyadicEP(1e-4, solver; state_ad=backend, parameter_ad=backend),
    )
    equilibrium = -(matrix \ bias)
    adjoint = transpose(matrix) \ (equilibrium .- target)
    @test gradient.bias ≈ -adjoint atol=2e-7
    @test gradient.weight ≈ -adjoint * transpose(equilibrium) atol=2e-7
    @test stats.converged

    asym_gradient, asym_stats = ep_gradient(
        problem,
        AsymEP(1e-4, solver; state_ad=backend, parameter_ad=backend),
    )
    @test asym_gradient.bias ≈ -adjoint atol=2e-7
    @test asym_gradient.weight ≈ -adjoint * transpose(equilibrium) atol=2e-7
    @test asym_stats.converged

    energy_layer = Lux.Dense(2 => 1; use_bias=false)
    energy_parameters, energy_state = lux_setup(rng, energy_layer)
    energy_parameters = merge(energy_parameters, (weight=reshape([0.3, -0.2], 1, 2),))
    energy_model = lux_energy_model(
        energy_layer;
        energy_output=(output, state, input) -> sum(state .^ 2) / 2 - only(output),
        cost=(state, y, ps, st) -> sum((state .- y) .^ 2) / 2,
        readout=(state, ps, st) -> state,
        initial_state=(ps, input, st) -> zeros(2),
    )
    energy_problem = EPProblem(energy_model, energy_parameters, energy_state, nothing, target)
    holomorphic_gradient, holomorphic_stats = ep_gradient(
        energy_problem,
        EPAlgorithm(HolomorphicEP(0.2; points=16), solver;
                    state_ad=backend, parameter_ad=backend),
    )
    @test vec(holomorphic_gradient.weight) ≈ [0.3, -0.2] .- target atol=2e-8
    @test holomorphic_stats.converged

    continuous = ContinuousEP(
        0.1,
        Relaxation(dt=0.1, maxiters=2, abstol=0.0);
        free_solver=solver,
        state_ad=backend,
        parameter_ad=backend,
        parameter_abstol=0.0,
        parameter_reltol=0.0,
    )
    optimizer_state = Optimisers.setup(Optimisers.Descent(1e-3), energy_parameters)
    _, continuous_parameters, continuous_stats = continuous_train_step!(
        optimizer_state, energy_problem, continuous,
    )
    @test continuous_stats.iterations == 2
    @test continuous_parameters != energy_parameters

    batchnorm = Lux.BatchNorm(2)
    batchnorm_parameters, _ = Lux.setup(rng, batchnorm)
    _, batchnorm_state = lux_setup(rng, batchnorm)
    batch_input = randn(rng, Float32, 2, 3)
    _, unchanged_state = Lux.apply(batchnorm, batch_input,
                                   batchnorm_parameters, batchnorm_state)
    @test isequal(unchanged_state, batchnorm_state)

    changing_layer = ChangingLuxLayer()
    changing_parameters, changing_state = Lux.setup(rng, changing_layer)
    changing_model = lux_dynamical_model(
        changing_layer;
        cost=(state, y, ps, st) -> sum(abs2, state .- y) / 2,
        readout=(state, ps, st) -> state,
        initial_state=(ps, input, st) -> zero(input),
    )
    @test_throws ArgumentError vector_field(
        changing_model, batch_input, changing_parameters, changing_state, nothing,
    )
end
