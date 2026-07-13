_phase_beta(::FreePhase) = false
_phase_beta(phase::NudgedPhase) = phase.β

"""
    equilibrate(problem, phase, algorithm; state_ad, initial_state_override=nothing)

Find an equilibrium of the phase's augmented energy with `algorithm`. Supplying
`initial_state_override` warm-starts the solve; otherwise the model's
[`initial_state`](@ref) is used.
"""
function equilibrate(
    problem::EPProblem,
    phase::AbstractPhase,
    algorithm::Relaxation;
    state_ad,
    initial_state_override=nothing,
)
    _check_backend(state_ad, "state")
    β = _phase_beta(phase)
    state = initial_state_override === nothing ?
        initial_state(problem.model, problem.parameters, problem.model_state, problem.input) :
        copy(initial_state_override)

    objective, preparation = _prepare_state_gradient(problem, state, β, state_ad)
    state_gradient = _state_gradient(objective, preparation, state_ad, state)
    residual = _tree_norm(state_gradient)
    initial_residual = residual
    tolerance = algorithm.abstol + algorithm.reltol * initial_residual
    iterations = 0

    while residual > tolerance && iterations < algorithm.maxiters
        state = fmap((x, g) -> x .- algorithm.dt .* g, state, state_gradient)
        iterations += 1
        state_gradient = _state_gradient(objective, preparation, state_ad, state)
        residual = _tree_norm(state_gradient)
    end

    value = augmented_energy(problem, state, β)
    return EquilibriumSolution(state, residual, iterations, residual <= tolerance, value, phase)
end

function CommonSolve.solve(problem::EPProblem, algorithm::Relaxation; state_ad, kwargs...)
    return equilibrate(problem, FreePhase(), algorithm; state_ad, kwargs...)
end


function CommonSolve.solve(
    problem::EPProblem,
    algorithm::AbstractEquilibriumAlgorithm;
    state_ad,
    kwargs...,
)
    return equilibrate(problem, FreePhase(), algorithm; state_ad, kwargs...)
end

function equilibrate(
    problem::EPProblem,
    phase::AbstractPhase,
    algorithm::Union{ODERelaxation,SteadyStateRelaxation,RootRelaxation};
    kwargs...,
)
    return _equilibrate_optional(problem, phase, algorithm; kwargs...)
end

function _equilibrate_optional(
    problem::EPProblem,
    phase::AbstractPhase,
    algorithm::Union{ODERelaxation,SteadyStateRelaxation,RootRelaxation};
    kwargs...,
)
    throw(ArgumentError(
        "$(nameof(typeof(algorithm))) requires its optional SciML integration packages; " *
        "load the package providing $(typeof(algorithm.algorithm)) before solving",
    ))
end
