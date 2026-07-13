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
    offset = Ref(0)
    return _rebuild_parameter_tree(template, packed, offset, copy_leaves)
end

function _rebuild_parameter_tree(template::AbstractArray, packed, offset, copy_leaves)
    count = length(template)
    indices = (offset[] + 1):(offset[] + count)
    offset[] += count
    result = reshape(view(packed, indices), size(template))
    return copy_leaves ? copy(result) : result
end

function _rebuild_parameter_tree(template::Number, packed, offset, copy_leaves)
    offset[] += 1
    return packed[offset[]]
end

function _rebuild_parameter_tree(template::NamedTuple, packed, offset, copy_leaves)
    rebuilt = map(
        value -> _rebuild_parameter_tree(value, packed, offset, copy_leaves),
        values(template),
    )
    return NamedTuple{keys(template)}(rebuilt)
end

function _rebuild_parameter_tree(template::Tuple, packed, offset, copy_leaves)
    return map(
        value -> _rebuild_parameter_tree(value, packed, offset, copy_leaves),
        template,
    )
end
