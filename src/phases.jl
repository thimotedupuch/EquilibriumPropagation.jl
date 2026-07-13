"""
    solve_phases(problem, algorithm)

Solve the free and required nudged phases. Nudged phases are warm-started from the
free equilibrium. Returns an [`EPPhases`](@ref) value.
"""
function solve_phases(problem::EPProblem, algorithm::EPAlgorithm)
    free = equilibrate(
        problem,
        FreePhase(),
        algorithm.free_solver;
        state_ad=algorithm.state_ad,
    )
    β = algorithm.protocol.β
    positive = equilibrate(
        problem,
        NudgedPhase(β),
        algorithm.positive_solver;
        state_ad=algorithm.state_ad,
        initial_state_override=free.state,
    )
    negative = _negative_phase(problem, free, algorithm.protocol, algorithm)
    return EPPhases(free, positive, negative)
end

_negative_phase(problem, free, ::OneSidedEP, algorithm) = nothing

function _negative_phase(problem, free, protocol::SymmetricEP, algorithm)
    return equilibrate(
        problem,
        NudgedPhase(-protocol.β),
        algorithm.negative_solver;
        state_ad=algorithm.state_ad,
        initial_state_override=free.state,
    )
end
