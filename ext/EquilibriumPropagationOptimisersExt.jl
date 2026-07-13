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
    problem::EPProblem,
    algorithm::Union{AsymEP,DyadicEP},
)
    gradients, stats = ep_gradient(problem, algorithm)
    optimizer_state, parameters = apply_gradient!(
        optimizer_state, problem.parameters, gradients,
    )
    return optimizer_state, parameters, stats
end

function EquilibriumPropagation.continuous_train_step!(
    optimizer_state,
    problem::EPProblem,
    algorithm::ContinuousEP,
)
    current_optimizer_state = optimizer_state
    function update_parameters(parameters, gradient, iteration)
        current_optimizer_state, parameters = apply_gradient!(
            current_optimizer_state,
            parameters,
            gradient,
        )
        return parameters
    end
    parameters, stats = continuous_ep(
        problem,
        algorithm;
        update_parameters,
    )
    return current_optimizer_state, parameters, stats
end

function EquilibriumPropagation.continuous_train_step!(
    optimizer_state,
    parameters,
    model,
    batch::Tuple,
    algorithm::ContinuousEP;
    model_state=NamedTuple(),
)
    length(batch) == 2 || throw(ArgumentError("batch must be an (input, target) tuple"))
    input, target = batch
    problem = EPProblem(model, parameters, model_state, input, target)
    return continuous_train_step!(optimizer_state, problem, algorithm)
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

function EquilibriumPropagation.train_step!(
    optimizer_state,
    parameters,
    model,
    batch::Tuple,
    algorithm::Union{AsymEP,DyadicEP};
    model_state=NamedTuple(),
)
    length(batch) == 2 || throw(ArgumentError("batch must be an (input, target) tuple"))
    input, target = batch
    return train_step!(
        optimizer_state,
        EPProblem(model, parameters, model_state, input, target),
        algorithm,
    )
end

end
