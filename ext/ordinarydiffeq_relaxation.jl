function _ode_iterations(solution)
    return hasproperty(solution.stats, :naccept) ? solution.stats.naccept : length(solution.t) - 1
end

function _solve_ode_relaxation(
    problem::EPProblem,
    phase,
    algorithm::ODERelaxation;
    state_ad,
    initial_state_override=nothing,
)
    EquilibriumPropagation._check_backend(state_ad, "state")
    state = initial_state_override === nothing ?
        initial_state(problem.model, problem.parameters, problem.model_state, problem.input) :
        copy(initial_state_override)
    state isa AbstractArray || throw(ArgumentError(
        "ODERelaxation currently supports only array-valued states",
    ))

    β = EquilibriumPropagation._phase_beta(phase)
    state_gradient = EquilibriumPropagation._state_gradient(problem, state, β, state_ad)
    initial_residual = EquilibriumPropagation._tree_norm(state_gradient)
    tolerance = algorithm.abstol + algorithm.reltol * initial_residual
    if initial_residual <= tolerance
        value = augmented_energy(problem, state, β)
        detail = (retcode=nothing, stats=nothing, solution=nothing)
        return EquilibriumSolution(state, initial_residual, 0, true, value, phase, detail)
    end

    # Implicit ODE methods may evaluate this function with dual-valued states while
    # constructing a Jacobian, so a preparation tied to the initial eltype cannot be reused.
    dynamics = (current, parameters, time) -> .-EquilibriumPropagation._state_gradient(
        problem, current, β, state_ad,
    )
    ode_problem = SciMLBase.ODEProblem(dynamics, state, algorithm.tspan)
    condition = (current, time, integrator) -> EquilibriumPropagation._tree_norm(
        EquilibriumPropagation._state_gradient(problem, current, β, state_ad),
    ) <= tolerance
    callback = SciMLBase.DiscreteCallback(
        condition,
        SciMLBase.terminate!;
        save_positions=(false, true),
    )
    solve_kwargs = algorithm.kwargs
    if haskey(solve_kwargs, :callback)
        callback = SciMLBase.CallbackSet(callback, solve_kwargs.callback)
        solve_kwargs = Base.structdiff(solve_kwargs, (callback=nothing,))
    end
    solution = SciMLBase.solve(
        ode_problem,
        algorithm.algorithm;
        abstol=algorithm.abstol,
        reltol=algorithm.reltol,
        callback,
        solve_kwargs...,
    )

    state = last(solution.u)
    state_gradient = EquilibriumPropagation._state_gradient(problem, state, β, state_ad)
    residual = EquilibriumPropagation._tree_norm(state_gradient)
    converged = SciMLBase.successful_retcode(solution) && residual <= tolerance
    value = augmented_energy(problem, state, β)
    detail = (retcode=solution.retcode, stats=solution.stats, solution=solution)
    return EquilibriumSolution(
        state,
        residual,
        _ode_iterations(solution),
        converged,
        value,
        phase,
        detail,
    )
end

function EquilibriumPropagation._equilibrate_optional(
    problem::EPProblem,
    phase::EquilibriumPropagation.AbstractPhase,
    algorithm::ODERelaxation{A};
    kwargs...,
) where {A<:SupportedODEAlgorithm}
    return _solve_ode_relaxation(problem, phase, algorithm; kwargs...)
end
