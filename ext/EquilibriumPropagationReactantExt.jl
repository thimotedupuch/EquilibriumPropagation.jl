module EquilibriumPropagationReactantExt

using EquilibriumPropagation
import Enzyme
import Reactant

struct ReactantEPInputs{P,S,B,I}
    parameters::P
    model_state::S
    batch::B
    initial_state::I
end

struct ReactantEPKernel{M,A}
    model::M
    algorithm::A
end

struct ReactantEPExecutable{T,I}
    thunk::T
    inputs::I
end

function _compiled_augmented_energy(
    state,
    model,
    parameters,
    model_state,
    batch,
    β,
)
    return augmented_energy(
        model,
        state,
        parameters,
        model_state,
        batch.input,
        batch.target,
        β,
    )
end

function _compiled_parameter_energy(
    parameters,
    model,
    state,
    model_state,
    batch,
    β,
)
    return augmented_energy(
        model,
        state,
        parameters,
        model_state,
        batch.input,
        batch.target,
        β,
    )
end

function _compiled_state_gradient(state, model, parameters, model_state, batch, β)
    return Enzyme.gradient(
        Enzyme.Reverse,
        _compiled_augmented_energy,
        state,
        Enzyme.Const(model),
        Enzyme.Const(parameters),
        Enzyme.Const(model_state),
        Enzyme.Const(batch),
        Enzyme.Const(β),
    )[1]
end

function _compiled_parameter_gradient(model, state, parameters, model_state, batch, β)
    return Enzyme.gradient(
        Enzyme.Reverse,
        _compiled_parameter_energy,
        parameters,
        Enzyme.Const(model),
        Enzyme.Const(state),
        Enzyme.Const(model_state),
        Enzyme.Const(batch),
        Enzyme.Const(β),
    )[1]
end

function _compiled_relax(
    state,
    model,
    parameters,
    model_state,
    batch,
    β,
    dt,
    steps,
)
    for _ in 1:steps
        derivative = _compiled_state_gradient(
            state, model, parameters, model_state, batch, β,
        )
        state = state .- dt .* derivative
    end
    return state
end

function _compiled_gradient_difference(first, second, scale)
    return EquilibriumPropagation.fmap(
        (x, y) -> (x .- y) ./ scale, first, second,
    )
end

function _compiled_sgd(parameters, gradient, learning_rate)
    learning_rate === nothing && return parameters
    return EquilibriumPropagation.fmap(
        (parameter, derivative) -> parameter .- learning_rate .* derivative,
        parameters,
        gradient,
    )
end

function _compiled_residual(model, state, parameters, model_state, batch, β)
    derivative = _compiled_state_gradient(
        state, model, parameters, model_state, batch, β,
    )
    return sqrt(sum(abs2, derivative))
end

function (kernel::ReactantEPKernel)(parameters, model_state, batch, initial_state)
    model = kernel.model
    algorithm = kernel.algorithm
    protocol = algorithm.protocol
    free_state = _compiled_relax(
        initial_state,
        model,
        parameters,
        model_state,
        batch,
        false,
        algorithm.dt,
        algorithm.free_steps,
    )
    positive_state = _compiled_relax(
        free_state,
        model,
        parameters,
        model_state,
        batch,
        protocol.β,
        algorithm.dt,
        algorithm.nudged_steps,
    )
    positive_gradient = _compiled_parameter_gradient(
        model, positive_state, parameters, model_state, batch, protocol.β,
    )

    if protocol isa OneSidedEP
        free_gradient = _compiled_parameter_gradient(
            model, free_state, parameters, model_state, batch, false,
        )
        gradient = _compiled_gradient_difference(
            positive_gradient, free_gradient, protocol.β,
        )
        negative_state = free_state
        negative_residual = zero(_compiled_residual(
            model, free_state, parameters, model_state, batch, false,
        ))
    else
        negative_state = _compiled_relax(
            free_state,
            model,
            parameters,
            model_state,
            batch,
            -protocol.β,
            algorithm.dt,
            algorithm.nudged_steps,
        )
        negative_gradient = _compiled_parameter_gradient(
            model, negative_state, parameters, model_state, batch, -protocol.β,
        )
        gradient = _compiled_gradient_difference(
            positive_gradient, negative_gradient, 2 * protocol.β,
        )
        negative_residual = _compiled_residual(
            model, negative_state, parameters, model_state, batch, -protocol.β,
        )
    end

    updated_parameters = _compiled_sgd(
        parameters, gradient, algorithm.learning_rate,
    )
    return (
        gradient=gradient,
        parameters=updated_parameters,
        free_state=free_state,
        positive_state=positive_state,
        negative_state=negative_state,
        loss=cost(model, free_state, parameters, model_state, batch.target),
        free_residual=_compiled_residual(
            model, free_state, parameters, model_state, batch, false,
        ),
        positive_residual=_compiled_residual(
            model, positive_state, parameters, model_state, batch, protocol.β,
        ),
        negative_residual=negative_residual,
    )
end

function _select_backend(backend)
    backend === nothing || Reactant.set_default_backend(backend)
    return nothing
end

function EquilibriumPropagation.reactant_inputs(
    problem::EPProblem{<:EPModel};
    backend=nothing,
)
    _select_backend(backend)
    initial = initial_state(
        problem.model,
        problem.parameters,
        problem.model_state,
        problem.input,
    )
    initial isa AbstractArray || throw(ArgumentError(
        "ReactantEP currently requires an array-valued dynamical state",
    ))
    batch = EPBatch(
        Reactant.to_rarray(problem.input; track_numbers=Number),
        Reactant.to_rarray(problem.target; track_numbers=Number),
    )
    return ReactantEPInputs(
        Reactant.to_rarray(problem.parameters; track_numbers=Number),
        Reactant.to_rarray(problem.model_state; track_numbers=Number),
        batch,
        Reactant.to_rarray(initial; track_numbers=Number),
    )
end

function EquilibriumPropagation.compile_reactant(
    problem::EPProblem{<:EPModel},
    algorithm::ReactantEP;
    backend=nothing,
    kwargs...,
)
    inputs = reactant_inputs(problem; backend)
    kernel = ReactantEPKernel(problem.model, algorithm)
    arguments = (
        inputs.parameters,
        inputs.model_state,
        inputs.batch,
        inputs.initial_state,
    )
    thunk = Reactant.compile(kernel, arguments; kwargs...)
    return ReactantEPExecutable(thunk, inputs)
end

function (executable::ReactantEPExecutable)(inputs::ReactantEPInputs=executable.inputs)
    return executable.thunk(
        inputs.parameters,
        inputs.model_state,
        inputs.batch,
        inputs.initial_state,
    )
end

function (executable::ReactantEPExecutable)(
    parameters,
    model_state,
    batch::EPBatch,
    initial_state,
)
    return executable.thunk(parameters, model_state, batch, initial_state)
end

end
