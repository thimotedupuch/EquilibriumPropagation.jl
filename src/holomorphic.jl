function _holomorphic_state_gradient(objective, state::AbstractArray, backend)
    eltype(state) <: Complex || throw(ArgumentError(
        "Holomorphic EP requires a complex-valued nudged state",
    ))
    real_state = real.(state)
    imaginary_state = imag.(state)
    lift(values) = complex.(values, imaginary_state)
    real_gradient = gradient(values -> real(objective(lift(values))), backend, real_state)
    imaginary_gradient = gradient(values -> imag(objective(lift(values))), backend, real_state)
    return complex.(real_gradient, imaginary_gradient)
end

function _complex_parameter_gradient(objective, parameters, backend)
    real_gradient = _parameter_objective_gradient(p -> real(objective(p)), parameters, backend)
    imaginary_gradient = _parameter_objective_gradient(p -> imag(objective(p)), parameters, backend)
    return fmap((real_part, imaginary_part) -> complex.(real_part, imaginary_part),
                real_gradient, imaginary_gradient)
end

function _equilibrate_holomorphic(
    problem::EPProblem{<:EPModel},
    β::Complex,
    solver::Relaxation,
    state_ad;
    initial_state_override,
)
    state = complex.(initial_state_override)
    objective = s -> augmented_energy(problem, s, β)
    state_gradient = _holomorphic_state_gradient(objective, state, state_ad)
    residual = _tree_norm(state_gradient)
    initial_residual = residual
    tolerance = solver.abstol + solver.reltol * initial_residual
    iterations = 0
    while residual > tolerance && iterations < solver.maxiters
        state = state .- solver.dt .* state_gradient
        iterations += 1
        state_gradient = _holomorphic_state_gradient(objective, state, state_ad)
        residual = _tree_norm(state_gradient)
    end
    return EquilibriumSolution(
        state,
        residual,
        iterations,
        residual <= tolerance,
        objective(state),
        NudgedPhase(β),
    )
end

function _holomorphic_phases(problem::EPProblem{<:EPModel}, algorithm::EPAlgorithm)
    protocol = algorithm.protocol
    protocol isa HolomorphicEP || throw(ArgumentError("expected a HolomorphicEP protocol"))
    free = equilibrate(
        problem,
        FreePhase(),
        algorithm.free_solver;
        state_ad=algorithm.state_ad,
    )
    free.state isa AbstractArray || throw(ArgumentError(
        "Holomorphic EP currently requires an array-valued dynamical state",
    ))
    angles = 2π .* (0:(protocol.points - 1)) ./ protocol.points
    nudges = protocol.radius .* cis.(angles)
    state = complex.(free.state)
    first_solution = _equilibrate_holomorphic(
        problem,
        first(nudges),
        algorithm.positive_solver,
        algorithm.state_ad;
        initial_state_override=state,
    )
    solutions = [first_solution]
    state = first_solution.state
    for index in 2:length(nudges)
        solution = _equilibrate_holomorphic(
            problem,
            nudges[index],
            algorithm.positive_solver,
            algorithm.state_ad;
            initial_state_override=state,
        )
        push!(solutions, solution)
        state = solution.state
    end
    return HolomorphicPhases(free, nudges, solutions)
end

function solve_phases(
    problem::EPProblem{<:EPModel},
    algorithm::EPAlgorithm{<:HolomorphicEP},
)
    _check_backend(algorithm.state_ad, "state")
    return _holomorphic_phases(problem, algorithm)
end

function _phase_parameter_gradient(problem, solution, parameter_ad)
    β = solution.phase.β
    objective = parameters -> augmented_energy(
        problem.model,
        solution.state,
        parameters,
        problem.model_state,
        problem.input,
        problem.target,
        β,
    )
    return _complex_parameter_gradient(objective, problem.parameters, parameter_ad)
end

"""
    ep_gradient(problem, algorithm::EPAlgorithm{<:HolomorphicEP}) -> gradients, stats

Solve a free phase and a circular sequence of complex nudged equilibria, then extract
the first Fourier coefficient of the phase-wise parameter derivatives. Parameters are
treated as real trainable values; the returned gradient is real and
`stats.imaginary_leakage` reports the discarded imaginary numerical residual.
"""
function ep_gradient(
    problem::EPProblem{<:EPModel},
    algorithm::EPAlgorithm{<:HolomorphicEP},
)
    _check_backend(algorithm.state_ad, "state")
    _check_backend(algorithm.parameter_ad, "parameter")
    phases = solve_phases(problem, algorithm)
    protocol = algorithm.protocol
    coefficient = nothing
    for index in eachindex(phases.solutions)
        phase_gradient = _phase_parameter_gradient(
            problem, phases.solutions[index], algorithm.parameter_ad,
        )
        weight = cis(-2π * (index - 1) / protocol.points)
        term = fmap(value -> value .* weight, phase_gradient)
        coefficient = coefficient === nothing ? term :
            fmap(+, coefficient, term)
    end
    estimate = fmap(value -> value ./ (protocol.points * protocol.radius), coefficient)
    gradients = fmap(value -> real.(value), estimate)
    leakage = _tree_norm(fmap(value -> imag.(value), estimate))
    displacements = map(
        solution -> _tree_distance(solution.state, phases.free.state),
        phases.solutions,
    )
    stats = HolomorphicEPStats(
        phases,
        cost(problem.model, phases.free.state, problem.parameters,
             problem.model_state, problem.target),
        _tree_norm(gradients),
        leakage,
        displacements,
        phases.free.converged && all(solution -> solution.converged, phases.solutions),
    )
    return gradients, stats
end
