"""
    augmented_energy(model, state, parameters, model_state, input, target, β)
    augmented_energy(problem, state, β)

Evaluate `energy + β * cost`, the scalar objective relaxed during an EP phase.
"""
function augmented_energy(model, state, parameters, model_state, input, target, β)
    return energy(model, state, parameters, model_state, input) +
           β * cost(model, state, parameters, model_state, target)
end

function augmented_energy(problem::EPProblem, state, β)
    return augmented_energy(
        problem.model,
        state,
        problem.parameters,
        problem.model_state,
        problem.input,
        problem.target,
        β,
    )
end
