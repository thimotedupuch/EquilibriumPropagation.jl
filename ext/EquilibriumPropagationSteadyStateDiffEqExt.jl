module EquilibriumPropagationSteadyStateDiffEqExt

using EquilibriumPropagation
import SciMLBase
import SteadyStateDiffEq

function EquilibriumPropagation._equilibrate_optional(
    problem::EPProblem,
    phase::EquilibriumPropagation.AbstractPhase,
    algorithm::SteadyStateRelaxation{A};
    state_ad,
    initial_state_override=nothing,
) where {A<:SciMLBase.AbstractODEAlgorithm}
    EquilibriumPropagation._check_backend(state_ad, "state")
    state = initial_state_override === nothing ?
        initial_state(problem.model, problem.parameters, problem.model_state, problem.input) :
        copy(initial_state_override)
    state isa AbstractArray || throw(ArgumentError(
        "SteadyStateRelaxation currently supports only array-valued states",
    ))

    β = EquilibriumPropagation._phase_beta(phase)
    state_gradient = EquilibriumPropagation._state_gradient(problem, state, β, state_ad)
    initial_residual = EquilibriumPropagation._tree_norm(state_gradient)
    tolerance = algorithm.abstol + algorithm.reltol * initial_residual
    if initial_residual <= tolerance
        value = augmented_energy(problem, state, β)
        detail = (retcode=nothing, stats=nothing, solution=nothing)
        return EquilibriumSolution(state, initial_residual, 0, true, value, phase, detail)
    end

    dynamics = (current, parameters, time) -> .-EquilibriumPropagation._state_gradient(
        problem, current, β, state_ad,
    )
    steady_problem = SciMLBase.SteadyStateProblem(dynamics, state)
    steady_algorithm = SteadyStateDiffEq.DynamicSS(
        algorithm.algorithm; tspan=algorithm.tspan,
    )
    solution = SciMLBase.solve(
        steady_problem,
        steady_algorithm;
        abstol=algorithm.abstol,
        reltol=algorithm.reltol,
        algorithm.kwargs...,
    )

    state = solution.u
    state_gradient = EquilibriumPropagation._state_gradient(problem, state, β, state_ad)
    residual = EquilibriumPropagation._tree_norm(state_gradient)
    converged = SciMLBase.successful_retcode(solution) && residual <= tolerance
    iterations = hasproperty(solution.stats, :naccept) ? solution.stats.naccept : 0
    value = augmented_energy(problem, state, β)
    detail = (retcode=solution.retcode, stats=solution.stats, solution=solution)
    return EquilibriumSolution(
        state, residual, iterations, converged, value, phase, detail,
    )
end

end
