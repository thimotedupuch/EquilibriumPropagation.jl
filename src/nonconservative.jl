function _require_array_state(state)
    state isa AbstractArray || throw(ArgumentError(
        "AsymEP and DyadicEP currently require an array-valued dynamical state",
    ))
    eltype(state) <: Number || throw(ArgumentError("dynamical state must be numeric"))
    return nothing
end

function _vector_field(problem::EPProblem{<:DynamicalModel}, state)
    return vector_field(problem.model, state, problem.parameters,
                        problem.model_state, problem.input)
end

function _cost_state_gradient(problem::EPProblem{<:DynamicalModel}, state, backend)
    objective = s -> cost(problem.model, s, problem.parameters,
                          problem.model_state, problem.target)
    return gradient(objective, backend, state)
end

_field_jacobian(problem::EPProblem{<:DynamicalModel}, state, backend) =
    jacobian(s -> _vector_field(problem, s), backend, state)

function _equilibrate_field(problem::EPProblem{<:DynamicalModel}, solver::Relaxation;
                            initial_state_override=nothing)
    state = initial_state_override === nothing ?
        initial_state(problem.model, problem.parameters, problem.model_state, problem.input) :
        copy(initial_state_override)
    _require_array_state(state)
    field = _vector_field(problem, state)
    size(field) == size(state) || throw(DimensionMismatch(
        "vector field size $(size(field)) must match state size $(size(state))",
    ))
    residual = _tree_norm(field)
    initial_residual = residual
    tolerance = solver.abstol + solver.reltol * initial_residual
    iterations = 0
    while residual > tolerance && iterations < solver.maxiters
        state = state .+ solver.dt .* field
        iterations += 1
        field = _vector_field(problem, state)
        residual = _tree_norm(field)
    end
    return EquilibriumSolution(state, residual, iterations, residual <= tolerance,
                               missing, FreePhase())
end

function equilibrate(
    problem::EPProblem{<:DynamicalModel},
    ::FreePhase,
    solver::Relaxation;
    state_ad=nothing,
    initial_state_override=nothing,
)
    return _equilibrate_field(
        problem, solver; initial_state_override,
    )
end

function _equilibrate_asym(problem::EPProblem{<:DynamicalModel}, free_state,
                           antisymmetric_jacobian, β, solver::Relaxation, state_ad)
    state = copy(free_state)
    function augmented_field(s)
        force = _vector_field(problem, s)
        cost_gradient = _cost_state_gradient(problem, s, state_ad)
        correction = reshape(
            antisymmetric_jacobian * vec(s .- free_state), size(s),
        )
        return force .- β .* cost_gradient .- 2 .* correction
    end
    field = augmented_field(state)
    residual = _tree_norm(field)
    initial_residual = residual
    tolerance = solver.abstol + solver.reltol * initial_residual
    iterations = 0
    while residual > tolerance && iterations < solver.maxiters
        state = state .+ solver.dt .* field
        iterations += 1
        field = augmented_field(state)
        residual = _tree_norm(field)
    end
    return EquilibriumSolution(state, residual, iterations, residual <= tolerance,
                               missing, NudgedPhase(β))
end

function _nonconservative_parameter_gradient(problem, state, signal, parameter_ad)
    objective = parameters -> begin
        supervised_cost = cost(problem.model, state, parameters,
                               problem.model_state, problem.target)
        force = vector_field(problem.model, state, parameters,
                             problem.model_state, problem.input)
        return supervised_cost - dot(vec(signal), vec(force))
    end
    return _parameter_objective_gradient(objective, problem.parameters, parameter_ad)
end

"""
    ep_gradient(problem, algorithm::AsymEP) -> gradients, stats

Solve the free and two antisymmetry-corrected nudged phases of a
[`DynamicalModel`](@ref), returning the equilibrium cost-gradient estimate and
[`NonConservativeStats`](@ref).
"""
function ep_gradient(problem::EPProblem{<:DynamicalModel}, algorithm::AsymEP)
    _check_backend(algorithm.state_ad, "state")
    _check_backend(algorithm.parameter_ad, "parameter")
    free = _equilibrate_field(problem, algorithm.free_solver)
    jac = _field_jacobian(problem, free.state, algorithm.state_ad)
    antisymmetric = (jac .- transpose(jac)) ./ 2
    plus = _equilibrate_asym(problem, free.state, antisymmetric, algorithm.β,
                             algorithm.positive_solver, algorithm.state_ad)
    minus = _equilibrate_asym(problem, free.state, antisymmetric, -algorithm.β,
                              algorithm.negative_solver, algorithm.state_ad)
    signal = (plus.state .- minus.state) ./ (2 * algorithm.β)
    gradients = _nonconservative_parameter_gradient(
        problem, free.state, signal, algorithm.parameter_ad,
    )
    stats = NonConservativeStats(
        :AsymEP, free, plus, minus,
        cost(problem.model, free.state, problem.parameters,
             problem.model_state, problem.target),
        _tree_norm(gradients), _tree_norm(antisymmetric),
        (positive=_tree_distance(plus.state, free.state),
         negative=_tree_distance(minus.state, free.state)),
        free.converged && plus.converged && minus.converged,
    )
    return gradients, stats
end

function _equilibrate_dyadic(problem::EPProblem{<:DynamicalModel}, free_state, β,
                             solver::Relaxation, state_ad)
    midpoint = copy(free_state)
    difference = zero(free_state)
    function rates(m, d)
        force = _vector_field(problem, m)
        jac = _field_jacobian(problem, m, state_ad)
        cost_gradient = _cost_state_gradient(problem, m, state_ad)
        difference_rate = reshape(transpose(jac) * vec(d), size(d)) .-
                          β .* cost_gradient
        return force, difference_rate
    end
    midpoint_rate, difference_rate = rates(midpoint, difference)
    residual = hypot(_tree_norm(midpoint_rate), _tree_norm(difference_rate))
    initial_residual = residual
    tolerance = solver.abstol + solver.reltol * initial_residual
    iterations = 0
    while residual > tolerance && iterations < solver.maxiters
        midpoint = midpoint .+ solver.dt .* midpoint_rate
        difference = difference .+ solver.dt .* difference_rate
        iterations += 1
        midpoint_rate, difference_rate = rates(midpoint, difference)
        residual = hypot(_tree_norm(midpoint_rate), _tree_norm(difference_rate))
    end
    state = (midpoint=midpoint, difference=difference)
    return EquilibriumSolution(state, residual, iterations, residual <= tolerance,
                               missing, NudgedPhase(β))
end

"""
    ep_gradient(problem, algorithm::DyadicEP) -> gradients, stats

Solve the free phase and doubled-state saddle dynamics of a [`DynamicalModel`](@ref),
returning the equilibrium cost-gradient estimate and [`NonConservativeStats`](@ref).
"""
function ep_gradient(problem::EPProblem{<:DynamicalModel}, algorithm::DyadicEP)
    _check_backend(algorithm.state_ad, "state")
    _check_backend(algorithm.parameter_ad, "parameter")
    free = _equilibrate_field(problem, algorithm.free_solver)
    nudged = _equilibrate_dyadic(problem, free.state, algorithm.β,
                                 algorithm.nudged_solver, algorithm.state_ad)
    signal = nudged.state.difference ./ algorithm.β
    gradients = _nonconservative_parameter_gradient(
        problem, nudged.state.midpoint, signal, algorithm.parameter_ad,
    )
    jac = _field_jacobian(problem, free.state, algorithm.state_ad)
    antisymmetric = (jac .- transpose(jac)) ./ 2
    stats = NonConservativeStats(
        :DyadicEP, free, nudged, nothing,
        cost(problem.model, free.state, problem.parameters,
             problem.model_state, problem.target),
        _tree_norm(gradients), _tree_norm(antisymmetric),
        _tree_distance(nudged.state.midpoint, free.state),
        free.converged && nudged.converged,
    )
    return gradients, stats
end
