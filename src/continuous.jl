function _tree_isfinite(tree)
    return all(fleaves(tree)) do leaf
        leaf isa Tuple{} || (leaf isa Number ? isfinite(leaf) : all(isfinite, leaf))
    end
end

_scale_gradient(gradient, scale) = fmap(value -> scale .* value, gradient)

function _scheduled_nudging(algorithm::ContinuousEP, iteration)
    β = algorithm.nudging_schedule(iteration, algorithm.β)
    β isa Number || throw(ArgumentError("nudging_schedule must return a number"))
    iszero(β) && throw(ArgumentError("nudging_schedule must return nonzero values"))
    sign(β) == sign(algorithm.β) || throw(ArgumentError(
        "nudging_schedule must preserve the sign of β",
    ))
    isfinite(β) || throw(ArgumentError("nudging_schedule must return finite values"))
    return β
end

function _scheduled_learning_scale(algorithm::ContinuousEP, iteration)
    scale = algorithm.learning_rate_schedule(iteration)
    scale isa Number || throw(ArgumentError(
        "learning_rate_schedule must return a number",
    ))
    scale >= zero(scale) || throw(ArgumentError(
        "learning_rate_schedule must return nonnegative values",
    ))
    isfinite(scale) || throw(ArgumentError(
        "learning_rate_schedule must return finite values",
    ))
    return scale
end

"""
    continuous_ep(problem, algorithm; update_parameters) -> parameters, stats

Run continual-update EP. `update_parameters(parameters, gradient, iteration)` is
called after every nudged state step and must return the parameters used by the next
state step. This optimizer-neutral callback receives a local gradient already scaled
by `algorithm.learning_rate_schedule`.

The first local difference contrasts the first nudged state with the free endpoint.
Later differences contrast consecutive nudged states, always differentiating with
respect to the parameters current at that iteration. This is the telescoping C-EP
rule; when parameter updates are slow, its sum approaches the ordinary one-sided EP
gradient.
"""
function continuous_ep(
    problem::EPProblem,
    algorithm::ContinuousEP;
    update_parameters,
)
    _check_backend(algorithm.state_ad, "state")
    _check_backend(algorithm.parameter_ad, "parameter")

    free = equilibrate(
        problem,
        FreePhase(),
        algorithm.free_solver;
        state_ad=algorithm.state_ad,
    )
    original_parameters = deepcopy(problem.parameters)
    parameters = problem.parameters
    state = copy(free.state)
    loss = cost(
        problem.model,
        free.state,
        parameters,
        problem.model_state,
        problem.target,
    )

    β = _scheduled_nudging(algorithm, 1)
    current_problem = EPProblem(
        problem.model,
        parameters,
        problem.model_state,
        problem.input,
        problem.target,
    )
    state_gradient = _state_gradient(current_problem, state, β, algorithm.state_ad)
    initial_residual = _tree_norm(state_gradient)
    state_tolerance = algorithm.nudged_solver.abstol +
                      algorithm.nudged_solver.reltol * initial_residual
    residual = initial_residual
    parameter_motion = zero(initial_residual)
    initial_parameter_motion = nothing
    state_converged = residual <= state_tolerance
    parameters_stable = algorithm.nudged_solver.maxiters == 0
    residual_history = typeof(initial_residual)[]
    parameter_motion_history = typeof(initial_residual)[]
    previous_β = false
    iterations = 0

    while iterations < algorithm.nudged_solver.maxiters &&
          !(state_converged && parameters_stable)
        iteration = iterations + 1
        iteration > 1 && (β = _scheduled_nudging(algorithm, iteration))
        learning_scale = _scheduled_learning_scale(algorithm, iteration)
        current_problem = EPProblem(
            problem.model,
            parameters,
            problem.model_state,
            problem.input,
            problem.target,
        )

        state_gradient = _state_gradient(current_problem, state, β, algorithm.state_ad)
        next_state = fmap(
            (value, derivative) ->
                value .- algorithm.nudged_solver.dt .* derivative,
            state,
            state_gradient,
        )

        before = _parameter_gradient(
            current_problem,
            state,
            previous_β,
            algorithm.parameter_ad,
        )
        after = _parameter_gradient(
            current_problem,
            next_state,
            β,
            algorithm.parameter_ad,
        )
        local_gradient = _gradient_difference(after, before, β)
        scaled_gradient = _scale_gradient(local_gradient, learning_scale)
        parameters_before_update = deepcopy(parameters)
        next_parameters = update_parameters(parameters, scaled_gradient, iteration)
        parameter_motion = _tree_distance(next_parameters, parameters_before_update)
        _tree_isfinite(next_parameters) || break

        parameters = next_parameters
        state = next_state
        previous_β = β
        iterations = iteration
        updated_problem = EPProblem(
            problem.model,
            parameters,
            problem.model_state,
            problem.input,
            problem.target,
        )
        state_gradient = _state_gradient(updated_problem, state, β, algorithm.state_ad)
        residual = _tree_norm(state_gradient)
        push!(residual_history, residual)
        push!(parameter_motion_history, parameter_motion)

        initial_parameter_motion === nothing &&
            (initial_parameter_motion = parameter_motion)
        parameter_tolerance = algorithm.parameter_abstol +
            algorithm.parameter_reltol * initial_parameter_motion
        state_converged = isfinite(residual) && residual <= state_tolerance
        parameters_stable = isfinite(parameter_motion) &&
                            parameter_motion <= parameter_tolerance
    end

    final_problem = EPProblem(
        problem.model,
        parameters,
        problem.model_state,
        problem.input,
        problem.target,
    )
    value = augmented_energy(final_problem, state, β)
    converged = free.converged && state_converged && parameters_stable
    detail = (
        residual_history=residual_history,
        parameter_motion_history=parameter_motion_history,
    )
    nudged = EquilibriumSolution(
        state,
        residual,
        iterations,
        converged,
        value,
        NudgedPhase(β),
        detail,
    )
    stats = ContinuousEPStats(
        free,
        nudged,
        loss,
        residual,
        parameter_motion,
        _tree_distance(parameters, original_parameters),
        β,
        iterations,
        state_converged,
        parameters_stable,
        converged,
        residual_history,
        parameter_motion_history,
    )
    return parameters, stats
end

continuous_ep(update_parameters, problem::EPProblem, algorithm::ContinuousEP) =
    continuous_ep(problem, algorithm; update_parameters)
