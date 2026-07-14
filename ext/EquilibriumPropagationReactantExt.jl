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

struct ReactantAsymKernel{M,A}
    model::M
    algorithm::A
end

struct ReactantDyadicKernel{M,A}
    model::M
    algorithm::A
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
    abstol,
    reltol,
)
    derivative = _compiled_state_gradient(
        state, model, parameters, model_state, batch, β,
    )
    initial_residual = _compiled_tree_norm(derivative)
    tolerance = abstol + reltol * initial_residual
    iterations = zero(steps)
    use_tolerance = !iszero(abstol) || !iszero(reltol)
    if use_tolerance
        residual = initial_residual
        Reactant.@trace while (iterations < steps) & (residual > tolerance)
            state = EquilibriumPropagation.fmap(
                (value, gradient) -> value .- dt .* gradient,
                state, derivative,
            )
            iterations += 1
            derivative = _compiled_state_gradient(
                state, model, parameters, model_state, batch, β,
            )
            residual = _compiled_tree_norm(derivative)
        end
    else
        for _ in 1:steps
            state = EquilibriumPropagation.fmap(
                (value, gradient) -> value .- dt .* gradient,
                state, derivative,
            )
            iterations += 1
            derivative = _compiled_state_gradient(
                state, model, parameters, model_state, batch, β,
            )
        end
    end
    residual = _compiled_tree_norm(derivative)
    converged = use_tolerance ? residual <= tolerance : false
    return (state=state, residual=residual, iterations=iterations,
            converged=converged)
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

function _compiled_tree_norm(tree)
    total = sum(EquilibriumPropagation.fleaves(tree); init=0) do leaf
        leaf isa Tuple{} && return 0
        return leaf isa Number ? abs2(leaf) : sum(abs2, leaf)
    end
    return sqrt(total)
end

function _compiled_vector_field(state, model, parameters, model_state, batch)
    return vector_field(model, state, parameters, model_state, batch.input)
end

function _compiled_cost_state_gradient(state, model, parameters, model_state, batch)
    return Enzyme.gradient(
        Enzyme.Reverse,
        (s, m, ps, st, b) -> cost(m, s, ps, st, b.target),
        state,
        Enzyme.Const(model),
        Enzyme.Const(parameters),
        Enzyme.Const(model_state),
        Enzyme.Const(batch),
    )[1]
end

function _compiled_field_jacobian(state, model, parameters, model_state, batch)
    columns = map(Enzyme.onehot(state)) do direction
        return Enzyme.autodiff(
            Enzyme.Forward, _compiled_vector_field,
            Enzyme.Duplicated(state, direction), Enzyme.Const(model),
            Enzyme.Const(parameters), Enzyme.Const(model_state),
            Enzyme.Const(batch),
        )[1]
    end
    return reduce(hcat, columns)
end

function _compiled_field_relax(state, model, parameters, model_state, batch,
                               dt, steps, abstol, reltol)
    field = _compiled_vector_field(state, model, parameters, model_state, batch)
    initial_residual = _compiled_tree_norm(field)
    tolerance = abstol + reltol * initial_residual
    iterations = zero(steps)
    use_tolerance = !iszero(abstol) || !iszero(reltol)
    if use_tolerance
        residual = initial_residual
        Reactant.@trace while (iterations < steps) & (residual > tolerance)
            state = state .+ dt .* field
            iterations += 1
            field = _compiled_vector_field(
                state, model, parameters, model_state, batch,
            )
            residual = _compiled_tree_norm(field)
        end
    else
        for _ in 1:steps
            state = state .+ dt .* field
            iterations += 1
            field = _compiled_vector_field(
                state, model, parameters, model_state, batch,
            )
        end
    end
    residual = _compiled_tree_norm(field)
    return (state=state, residual=residual, iterations=iterations,
            converged=use_tolerance ? residual <= tolerance : false)
end

function _compiled_nonconservative_parameter_gradient(
    parameters, model, state, signal, model_state, batch,
)
    objective = function (ps, m, s, q, st, b)
        supervised = cost(m, s, ps, st, b.target)
        field = vector_field(m, s, ps, st, b.input)
        return supervised - sum(q .* field)
    end
    return Enzyme.gradient(
        Enzyme.Reverse, objective, parameters, Enzyme.Const(model),
        Enzyme.Const(state), Enzyme.Const(signal), Enzyme.Const(model_state),
        Enzyme.Const(batch),
    )[1]
end

function (kernel::ReactantAsymKernel)(parameters, model_state, batch, initial_state)
    model, algorithm = kernel.model, kernel.algorithm
    free = _compiled_field_relax(
        initial_state, model, parameters, model_state, batch,
        algorithm.free_solver.dt, algorithm.free_solver.maxiters,
        algorithm.free_solver.abstol, algorithm.free_solver.reltol,
    )
    jacobian = _compiled_field_jacobian(
        free.state, model, parameters, model_state, batch,
    )
    antisymmetric = (jacobian .- transpose(jacobian)) ./ 2
    function relax_nudged(β, solver)
        state = free.state
        function augmented_field(value)
            field = _compiled_vector_field(
                value, model, parameters, model_state, batch,
            )
            cost_gradient = _compiled_cost_state_gradient(
                value, model, parameters, model_state, batch,
            )
            correction = reshape(
                antisymmetric * vec(value .- free.state), size(value),
            )
            return field .- β .* cost_gradient .- 2 .* correction
        end
        field = augmented_field(state)
        initial_residual = _compiled_tree_norm(field)
        tolerance = solver.abstol + solver.reltol * initial_residual
        iterations = zero(solver.maxiters)
        use_tolerance = !iszero(solver.abstol) || !iszero(solver.reltol)
        for _ in 1:solver.maxiters
            active = use_tolerance ? _compiled_tree_norm(field) > tolerance : true
            state = ifelse.(active, state .+ solver.dt .* field, state)
            iterations += active
            field = augmented_field(state)
        end
        residual = _compiled_tree_norm(field)
        return (state=state, residual=residual, iterations=iterations,
                converged=use_tolerance ? residual <= tolerance : false)
    end
    positive = relax_nudged(algorithm.β, algorithm.positive_solver)
    negative = relax_nudged(-algorithm.β, algorithm.negative_solver)
    signal = (positive.state .- negative.state) ./ (2 * algorithm.β)
    gradient = _compiled_nonconservative_parameter_gradient(
        parameters, model, free.state, signal, model_state, batch,
    )
    return (
        gradient=gradient, parameters=parameters, free_state=free.state,
        positive_state=positive.state, negative_state=negative.state,
        loss=cost(model, free.state, parameters, model_state, batch.target),
        free_residual=free.residual, positive_residual=positive.residual,
        negative_residual=negative.residual, free_iterations=free.iterations,
        positive_iterations=positive.iterations,
        negative_iterations=negative.iterations,
        jacobian_asymmetry=_compiled_tree_norm(antisymmetric),
        converged=free.converged & positive.converged & negative.converged,
    )
end

function (kernel::ReactantDyadicKernel)(parameters, model_state, batch, initial_state)
    model, algorithm = kernel.model, kernel.algorithm
    free = _compiled_field_relax(
        initial_state, model, parameters, model_state, batch,
        algorithm.free_solver.dt, algorithm.free_solver.maxiters,
        algorithm.free_solver.abstol, algorithm.free_solver.reltol,
    )
    midpoint = free.state
    difference = zero(free.state)
    function rates(middle, delta)
        field = _compiled_vector_field(
            middle, model, parameters, model_state, batch,
        )
        jacobian = _compiled_field_jacobian(
            middle, model, parameters, model_state, batch,
        )
        cost_gradient = _compiled_cost_state_gradient(
            middle, model, parameters, model_state, batch,
        )
        difference_rate = reshape(
            transpose(jacobian) * vec(delta), size(delta),
        ) .- algorithm.β .* cost_gradient
        return field, difference_rate
    end
    midpoint_rate, difference_rate = rates(midpoint, difference)
    solver = algorithm.nudged_solver
    initial_residual = hypot(
        _compiled_tree_norm(midpoint_rate), _compiled_tree_norm(difference_rate),
    )
    tolerance = solver.abstol + solver.reltol * initial_residual
    iterations = zero(solver.maxiters)
    use_tolerance = !iszero(solver.abstol) || !iszero(solver.reltol)
    for _ in 1:solver.maxiters
        residual = hypot(
            _compiled_tree_norm(midpoint_rate),
            _compiled_tree_norm(difference_rate),
        )
        active = use_tolerance ? residual > tolerance : true
        midpoint = ifelse.(active, midpoint .+ solver.dt .* midpoint_rate, midpoint)
        difference = ifelse.(
            active, difference .+ solver.dt .* difference_rate, difference,
        )
        iterations += active
        midpoint_rate, difference_rate = rates(midpoint, difference)
    end
    residual = hypot(
        _compiled_tree_norm(midpoint_rate), _compiled_tree_norm(difference_rate),
    )
    signal = difference ./ algorithm.β
    gradient = _compiled_nonconservative_parameter_gradient(
        parameters, model, midpoint, signal, model_state, batch,
    )
    jacobian = _compiled_field_jacobian(
        free.state, model, parameters, model_state, batch,
    )
    antisymmetric = (jacobian .- transpose(jacobian)) ./ 2
    nudged_converged = use_tolerance ? residual <= tolerance : false
    return (
        gradient=gradient, parameters=parameters, free_state=free.state,
        midpoint_state=midpoint, difference_state=difference,
        loss=cost(model, free.state, parameters, model_state, batch.target),
        free_residual=free.residual, nudged_residual=residual,
        free_iterations=free.iterations, nudged_iterations=iterations,
        jacobian_asymmetry=_compiled_tree_norm(antisymmetric),
        converged=free.converged & nudged_converged,
    )
end

function (kernel::ReactantEPKernel)(parameters, model_state, batch, initial_state)
    model = kernel.model
    algorithm = kernel.algorithm
    protocol = algorithm.protocol
    free = _compiled_relax(
        initial_state,
        model,
        parameters,
        model_state,
        batch,
        false,
        algorithm.dt,
        algorithm.free_steps,
        algorithm.abstol,
        algorithm.reltol,
    )
    free_state = free.state
    positive = _compiled_relax(
        free_state,
        model,
        parameters,
        model_state,
        batch,
        protocol.β,
        algorithm.dt,
        algorithm.nudged_steps,
        algorithm.abstol,
        algorithm.reltol,
    )
    positive_state = positive.state
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
        negative_residual = zero(free.residual)
        negative_iterations = zero(free.iterations)
        negative_converged = true
    else
        negative = _compiled_relax(
            free_state,
            model,
            parameters,
            model_state,
            batch,
            -protocol.β,
            algorithm.dt,
            algorithm.nudged_steps,
            algorithm.abstol,
            algorithm.reltol,
        )
        negative_state = negative.state
        negative_gradient = _compiled_parameter_gradient(
            model, negative_state, parameters, model_state, batch, -protocol.β,
        )
        gradient = _compiled_gradient_difference(
            positive_gradient, negative_gradient, 2 * protocol.β,
        )
        negative_residual = negative.residual
        negative_iterations = negative.iterations
        negative_converged = negative.converged
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
        free_residual=free.residual,
        positive_residual=positive.residual,
        negative_residual=negative_residual,
        free_iterations=free.iterations,
        positive_iterations=positive.iterations,
        negative_iterations=negative_iterations,
        free_converged=free.converged,
        positive_converged=positive.converged,
        negative_converged=negative_converged,
        converged=free.converged & positive.converged & negative_converged,
    )
end

function _select_backend(backend)
    backend === nothing || Reactant.set_default_backend(backend)
    return nothing
end

function _reactant_batch(input, target)
    transferred_input = Reactant.to_rarray(input; track_numbers=Number)
    transferred_target = Reactant.to_rarray(target; track_numbers=Number)
    return EPBatch{typeof(transferred_input),typeof(transferred_target)}(
        transferred_input, transferred_target,
    )
end

function _reactant_inputs(problem, initial)
    return ReactantEPInputs(
        Reactant.to_rarray(problem.parameters; track_numbers=Number),
        Reactant.to_rarray(problem.model_state; track_numbers=Number),
        _reactant_batch(problem.input, problem.target),
        Reactant.to_rarray(initial; track_numbers=Number),
    )
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
    return _reactant_inputs(problem, initial)
end

function EquilibriumPropagation.reactant_inputs(
    problem::EPProblem{<:DynamicalModel}; backend=nothing,
)
    _select_backend(backend)
    initial = initial_state(
        problem.model, problem.parameters, problem.model_state, problem.input,
    )
    initial isa AbstractArray || throw(ArgumentError(
        "compiled AsymEP and DyadicEP currently require an array-valued state",
    ))
    return _reactant_inputs(problem, initial)
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

function EquilibriumPropagation.compile_reactant(
    problem::EPProblem{<:EPModel},
    algorithm::EPAlgorithm;
    backend=nothing,
    learning_rate=nothing,
    kwargs...,
)
    algorithm.protocol isa HolomorphicEP && throw(ArgumentError(
        "HolomorphicEP cannot currently be compiled safely: Reactant 0.2 leaves " *
        "complex Enzyme batch operations unlowered and its complex dot-general " *
        "optimization can crash natively",
    ))
    solvers = (
        algorithm.free_solver, algorithm.positive_solver, algorithm.negative_solver,
    )
    all(solver -> solver isa Relaxation, solvers) || throw(ArgumentError(
        "SciML-backed relaxation cannot be enclosed in a StableHLO executable: " *
        "its adaptive integrator orchestration, callbacks, and solver state run " *
        "on the Julia host; use Relaxation for whole-step Reactant compilation",
    ))
    free_solver, positive_solver, negative_solver = solvers
    positive_solver.dt == negative_solver.dt || throw(ArgumentError(
        "compiled positive and negative solvers must use the same dt",
    ))
    positive_solver.maxiters == negative_solver.maxiters || throw(ArgumentError(
        "compiled positive and negative solvers must use the same maxiters",
    ))
    free_solver.dt == positive_solver.dt || throw(ArgumentError(
        "compiled free and nudged solvers must use the same dt",
    ))
    free_solver.abstol == positive_solver.abstol == negative_solver.abstol ||
        throw(ArgumentError("compiled phase solvers must use the same abstol"))
    free_solver.reltol == positive_solver.reltol == negative_solver.reltol ||
        throw(ArgumentError("compiled phase solvers must use the same reltol"))
    compiled_algorithm = ReactantEP(
        algorithm.protocol; dt=free_solver.dt,
        free_steps=free_solver.maxiters,
        nudged_steps=positive_solver.maxiters,
        abstol=free_solver.abstol, reltol=free_solver.reltol,
        learning_rate,
    )
    return compile_reactant(problem, compiled_algorithm; backend, kwargs...)
end

function EquilibriumPropagation.compile_reactant(
    problem::EPProblem{<:EPModel},
    algorithm::ReactantEP{<:HolomorphicEP};
    backend=nothing,
    kwargs...,
)
    throw(ArgumentError(
        "HolomorphicEP cannot currently be compiled safely: Reactant 0.2 leaves " *
        "complex Enzyme batch operations unlowered, and its optimized complex " *
        "dot-general pass can crash natively",
    ))
end

function _compile_reactant_kernel(problem, algorithm, kernel; backend, kwargs)
    inputs = reactant_inputs(problem; backend)
    arguments = (
        inputs.parameters, inputs.model_state, inputs.batch, inputs.initial_state,
    )
    thunk = Reactant.compile(kernel, arguments; kwargs...)
    return ReactantEPExecutable(thunk, inputs)
end

function EquilibriumPropagation.compile_reactant(
    problem::EPProblem{<:DynamicalModel}, algorithm::AsymEP;
    backend=nothing, kwargs...,
)
    return _compile_reactant_kernel(
        problem, algorithm, ReactantAsymKernel(problem.model, algorithm);
        backend, kwargs,
    )
end

function EquilibriumPropagation.compile_reactant(
    problem::EPProblem{<:DynamicalModel}, algorithm::DyadicEP;
    backend=nothing, kwargs...,
)
    return _compile_reactant_kernel(
        problem, algorithm, ReactantDyadicKernel(problem.model, algorithm);
        backend, kwargs,
    )
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
