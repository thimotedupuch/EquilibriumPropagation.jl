module EquilibriumPropagationReactantOptimisersExt

using EquilibriumPropagation
import Enzyme
import Optimisers
import Reactant

const ReactantExt = Base.get_extension(
    EquilibriumPropagation, :EquilibriumPropagationReactantExt,
)

struct ReactantOptimiserInputs{P,S,B,I,O}
    parameters::P
    model_state::S
    batch::B
    initial_state::I
    optimizer_state::O
end

struct ReactantOptimiserExecutable{T,I}
    thunk::T
    inputs::I
end

struct OptimiserKernel{K}
    kernel::K
end

struct ContinuousOptimiserKernel{M,A}
    model::M
    algorithm::A
end

function (kernel::OptimiserKernel)(
    parameters, model_state, batch, initial_state, optimizer_state,
)
    result = kernel.kernel(parameters, model_state, batch, initial_state)
    optimizer_state, updated_parameters = Optimisers.update(
        optimizer_state, parameters, result.gradient,
    )
    return merge(result, (
        parameters=updated_parameters, optimizer_state=optimizer_state,
    ))
end

function _select_tree(condition, next, previous)
    return EquilibriumPropagation.fmap(
        (x, y) -> ifelse.(condition, x, y), next, previous,
    )
end

function (kernel::ContinuousOptimiserKernel)(
    parameters, model_state, batch, initial_state, optimizer_state,
)
    model, algorithm = kernel.model, kernel.algorithm
    free_solver = algorithm.free_solver
    free = ReactantExt._compiled_relax(
        initial_state, model, parameters, model_state, batch, false,
        free_solver.dt, free_solver.maxiters, free_solver.abstol,
        free_solver.reltol,
    )
    original_parameters = parameters
    state = free.state
    previous_β = false
    residual = free.residual
    parameter_motion = zero(residual)
    iterations = zero(algorithm.nudged_solver.maxiters)
    state_converged = false
    parameters_stable = false
    solver = algorithm.nudged_solver
    for iteration in 1:solver.maxiters
        β = algorithm.nudging_schedule(iteration, algorithm.β)
        learning_scale = algorithm.learning_rate_schedule(iteration)
        state_gradient = ReactantExt._compiled_state_gradient(
            state, model, parameters, model_state, batch, β,
        )
        next_state = EquilibriumPropagation.fmap(
            (value, derivative) -> value .- solver.dt .* derivative,
            state, state_gradient,
        )
        before = ReactantExt._compiled_parameter_gradient(
            model, state, parameters, model_state, batch, previous_β,
        )
        after = ReactantExt._compiled_parameter_gradient(
            model, next_state, parameters, model_state, batch, β,
        )
        local_gradient = ReactantExt._compiled_gradient_difference(
            after, before, β,
        )
        scaled_gradient = EquilibriumPropagation.fmap(
            value -> learning_scale .* value, local_gradient,
        )
        candidate_optimizer_state, candidate_parameters = Optimisers.update(
            optimizer_state, parameters, scaled_gradient,
        )
        parameter_motion = ReactantExt._compiled_tree_norm(
            EquilibriumPropagation.fmap(
                (next, current) -> next .- current,
                candidate_parameters, parameters,
            ),
        )
        parameters = candidate_parameters
        optimizer_state = candidate_optimizer_state
        state = next_state
        previous_β = β
        iterations += 1
        state_gradient = ReactantExt._compiled_state_gradient(
            state, model, parameters, model_state, batch, β,
        )
        residual = ReactantExt._compiled_tree_norm(state_gradient)
        state_tolerance = solver.abstol + solver.reltol * free.residual
        parameter_tolerance = algorithm.parameter_abstol +
                              algorithm.parameter_reltol * parameter_motion
        state_converged = residual <= state_tolerance
        parameters_stable = parameter_motion <= parameter_tolerance
    end
    displacement = ReactantExt._compiled_tree_norm(
        EquilibriumPropagation.fmap(
            (current, original) -> current .- original,
            parameters, original_parameters,
        ),
    )
    return (
        parameters=parameters, optimizer_state=optimizer_state,
        free_state=free.state, nudged_state=state,
        loss=cost(model, free.state, original_parameters, model_state, batch.target),
        free_residual=free.residual, nudged_residual=residual,
        parameter_motion=parameter_motion,
        parameter_displacement=displacement, iterations=iterations,
        state_converged=state_converged,
        parameters_stable=parameters_stable,
        converged=free.converged & state_converged & parameters_stable,
    )
end

function _base_kernel(problem, algorithm::ReactantEP)
    algorithm.learning_rate === nothing || throw(ArgumentError(
        "set ReactantEP learning_rate=nothing when supplying an Optimisers state",
    ))
    return ReactantExt.ReactantEPKernel(problem.model, algorithm)
end
_base_kernel(problem, algorithm::AsymEP) =
    ReactantExt.ReactantAsymKernel(problem.model, algorithm)
_base_kernel(problem, algorithm::DyadicEP) =
    ReactantExt.ReactantDyadicKernel(problem.model, algorithm)

function _compile_with_optimizer(
    problem, algorithm, optimizer, kernel; backend, kwargs,
)
    prepared = reactant_inputs(problem; backend)
    device_optimizer_state = if optimizer isa Optimisers.AbstractRule
        device_optimizer = Reactant.to_rarray(
            optimizer; track_numbers=AbstractFloat,
        )
        Reactant.@jit Optimisers.setup(device_optimizer, prepared.parameters)
    else
        try
            Reactant.to_rarray(optimizer; track_numbers=Number)
        catch error
            throw(ArgumentError(
                "pass an Optimisers.AbstractRule (for example Adam(1f-3)) or " *
                "an optimizer state already returned by a compiled executable; " *
                "host Optimisers.Leaf states cannot be structurally transferred",
            ))
        end
    end
    inputs = ReactantOptimiserInputs(
        prepared.parameters, prepared.model_state, prepared.batch,
        prepared.initial_state, device_optimizer_state,
    )
    arguments = (
        inputs.parameters, inputs.model_state, inputs.batch,
        inputs.initial_state, inputs.optimizer_state,
    )
    thunk = Reactant.compile(kernel, arguments; kwargs...)
    return ReactantOptimiserExecutable(thunk, inputs)
end

function EquilibriumPropagation.compile_reactant(
    problem::EPProblem,
    algorithm::Union{ReactantEP,AsymEP,DyadicEP},
    optimizer;
    backend=nothing,
    kwargs...,
)
    return _compile_with_optimizer(
        problem, algorithm, optimizer,
        OptimiserKernel(_base_kernel(problem, algorithm)); backend, kwargs,
    )
end

function EquilibriumPropagation.compile_reactant(
    problem::EPProblem{<:EPModel},
    algorithm::ContinuousEP,
    optimizer;
    backend=nothing,
    kwargs...,
)
    return _compile_with_optimizer(
        problem, algorithm, optimizer,
        ContinuousOptimiserKernel(problem.model, algorithm); backend, kwargs,
    )
end

function (executable::ReactantOptimiserExecutable)(
    inputs::ReactantOptimiserInputs=executable.inputs,
)
    return executable.thunk(
        inputs.parameters, inputs.model_state, inputs.batch,
        inputs.initial_state, inputs.optimizer_state,
    )
end

function (executable::ReactantOptimiserExecutable)(
    parameters, model_state, batch::EPBatch, initial_state, optimizer_state,
)
    return executable.thunk(
        parameters, model_state, batch, initial_state, optimizer_state,
    )
end

end
