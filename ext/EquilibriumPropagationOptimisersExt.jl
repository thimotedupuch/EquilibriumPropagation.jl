module EquilibriumPropagationOptimisersExt

using EquilibriumPropagation
import Optimisers

function EquilibriumPropagation.apply_gradient!(optimizer_state, parameters, gradients)
    return Optimisers.update(optimizer_state, parameters, gradients)
end

function EquilibriumPropagation.train_step!(
    optimizer_state,
    problem::EPProblem,
    algorithm::EPAlgorithm,
)
    gradients, stats = ep_gradient(problem, algorithm)
    optimizer_state, parameters = apply_gradient!(
        optimizer_state,
        problem.parameters,
        gradients,
    )
    return optimizer_state, parameters, stats
end

function EquilibriumPropagation.train_step!(
    optimizer_state,
    parameters,
    model,
    batch::Tuple,
    algorithm::EPAlgorithm;
    model_state=NamedTuple(),
)
    length(batch) == 2 || throw(ArgumentError("batch must be an (input, target) tuple"))
    input, target = batch
    problem = EPProblem(model, parameters, model_state, input, target)
    return train_step!(optimizer_state, problem, algorithm)
end

end
