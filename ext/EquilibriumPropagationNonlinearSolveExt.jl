module EquilibriumPropagationNonlinearSolveExt

using EquilibriumPropagation
import NonlinearSolve
import SciMLBase

function EquilibriumPropagation._equilibrate_optional(
    problem::EPProblem,
    phase::EquilibriumPropagation.AbstractPhase,
    algorithm::RootRelaxation{A};
    state_ad,
    initial_state_override=nothing,
) where {A<:SciMLBase.AbstractNonlinearAlgorithm}
    EquilibriumPropagation._check_backend(state_ad, "state")
    state = initial_state_override === nothing ?
        initial_state(problem.model, problem.parameters, problem.model_state, problem.input) :
        copy(initial_state_override)
    state isa AbstractArray || throw(ArgumentError(
        "RootRelaxation currently supports only array-valued states",
    ))

    β = EquilibriumPropagation._phase_beta(phase)
    equilibrium_equation = (current, parameters) -> EquilibriumPropagation._state_gradient(
        problem, current, β, state_ad,
    )
    initial_residual = EquilibriumPropagation._tree_norm(equilibrium_equation(state, nothing))
    tolerance = algorithm.abstol + algorithm.reltol * initial_residual
    if initial_residual <= tolerance
        value = augmented_energy(problem, state, β)
        detail = (retcode=nothing, stats=nothing, solution=nothing)
        return EquilibriumSolution(state, initial_residual, 0, true, value, phase, detail)
    end
    nonlinear_problem = SciMLBase.NonlinearProblem(equilibrium_equation, state)
    solution = SciMLBase.solve(
        nonlinear_problem,
        algorithm.algorithm;
        abstol=algorithm.abstol,
        reltol=algorithm.reltol,
        algorithm.kwargs...,
    )

    state = solution.u
    state_gradient = equilibrium_equation(state, nothing)
    residual = EquilibriumPropagation._tree_norm(state_gradient)
    converged = SciMLBase.successful_retcode(solution) && residual <= tolerance
    iterations = hasproperty(solution.stats, :nsteps) ? solution.stats.nsteps : 0
    value = augmented_energy(problem, state, β)
    detail = (retcode=solution.retcode, stats=solution.stats, solution=solution)
    return EquilibriumSolution(
        state, residual, iterations, converged, value, phase, detail,
    )
end

end
