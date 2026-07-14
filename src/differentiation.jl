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
    return _parameter_objective_gradient(objective, problem.parameters, backend)
end

function _parameter_objective_gradient(objective, parameters, backend)
    parameters isa AbstractArray && return gradient(objective, backend, parameters)

    # Array-oriented forward-mode backends cannot accept a parameter tree directly.
    # Flatten tuple/named-tuple trees only at the AD boundary, then restore both the
    # parameters passed to the model and the gradient returned to the caller.
    parameters isa Union{Tuple,NamedTuple} || return gradient(objective, backend, parameters)
    packed = _flatten_parameter_tree(parameters)
    packed_objective = values -> objective(_rebuild_parameter_tree(parameters, values))
    packed_gradient = gradient(packed_objective, backend, packed)
    return _rebuild_parameter_tree(parameters, packed_gradient; copy_leaves=true)
end

function _parameter_leaves!(leaves, value::AbstractArray)
    eltype(value) <: Number || throw(ArgumentError("parameter arrays must contain numbers"))
    push!(leaves, vec(value))
    return leaves
end

function _parameter_leaves!(leaves, value::Number)
    push!(leaves, [value])
    return leaves
end

function _parameter_leaves!(leaves, tree::Union{Tuple,NamedTuple})
    for value in values(tree)
        _parameter_leaves!(leaves, value)
    end
    return leaves
end

function _parameter_leaves!(leaves, value)
    throw(ArgumentError(
        "tuple parameter trees may only contain numeric arrays, numbers, and nested tuples; " *
        "found $(typeof(value))",
    ))
end

function _flatten_parameter_tree(tree)
    leaves = _parameter_leaves!(Any[], tree)
    isempty(leaves) && throw(ArgumentError("parameter tree contains no numeric leaves"))
    return reduce(vcat, leaves)
end

function _rebuild_parameter_tree(template, packed; copy_leaves=false)
    rebuilt, next_index = _rebuild_parameter_tree(template, packed, 1, copy_leaves)
    next_index == length(packed) + 1 || throw(DimensionMismatch(
        "packed parameter vector has $(length(packed)) entries, but the template consumes " *
        "$(next_index - 1)",
    ))
    return rebuilt
end

function _rebuild_parameter_tree(template::AbstractArray, packed, first_index, copy_leaves)
    count = length(template)
    indices = first_index:(first_index + count - 1)
    result = reshape(view(packed, indices), size(template))
    return (copy_leaves ? copy(result) : result), first_index + count
end

function _rebuild_parameter_tree(template::Number, packed, first_index, copy_leaves)
    return packed[first_index], first_index + 1
end

function _rebuild_parameter_tree(template::NamedTuple, packed, first_index, copy_leaves)
    rebuilt, next_index = _rebuild_parameter_tree(
        values(template), packed, first_index, copy_leaves,
    )
    return NamedTuple{keys(template)}(rebuilt), next_index
end

function _rebuild_parameter_tree(template::Tuple, packed, first_index, copy_leaves)
    isempty(template) && return (), first_index
    head, after_head = _rebuild_parameter_tree(
        first(template), packed, first_index, copy_leaves,
    )
    tail, next_index = _rebuild_parameter_tree(
        Base.tail(template), packed, after_head, copy_leaves,
    )
    return (head, tail...), next_index
end
