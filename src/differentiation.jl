function _check_backend(backend, purpose)
    available = try
        check_available(backend)
    catch error
        throw(ArgumentError("the $purpose AD backend is not available: $(sprint(showerror, error))"))
    end
    available || throw(ArgumentError(
        "the $purpose AD backend is not available; load its implementation package",
    ))
    return nothing
end

function _state_gradient(problem::EPProblem, state, β, backend)
    objective = s -> augmented_energy(problem, s, β)
    return gradient(objective, backend, state)
end

function _prepare_state_gradient(problem::EPProblem, state, β, backend)
    objective = s -> augmented_energy(problem, s, β)
    preparation = prepare_gradient(objective, backend, state)
    return objective, preparation
end

function _state_gradient(objective, preparation, backend, state)
    return gradient(objective, preparation, backend, state)
end

function _parameter_gradient(problem::EPProblem, state, β, backend)
    # `state` belongs to a completed phase and is deliberately closed over here.
    # Only `parameters` is active, so no derivative can pass through equilibrium solving.
    objective = parameters -> augmented_energy(
        problem.model,
        state,
        parameters,
        problem.model_state,
        problem.input,
        problem.target,
        β,
    )
    return gradient(objective, backend, problem.parameters)
end
