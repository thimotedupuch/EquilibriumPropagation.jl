function _gradient_difference(a, b, scale)
    return fmap((x, y) -> (x .- y) ./ scale, a, b)
end

"""
    ep_gradient(problem, phases, algorithm)
    ep_gradient(problem, algorithm) -> gradients, stats

Construct the one-sided or symmetric EP parameter gradient. Equilibrated phase states
are held constant during parameter differentiation, so this operation never
differentiates through the equilibrium solver.

The two-argument form solves all phases and also returns [`EPStats`](@ref).
"""
function ep_gradient(
    problem::EPProblem,
    phases::EPPhases,
    protocol::OneSidedEP,
    parameter_ad,
)
    _check_backend(parameter_ad, "parameter")
    nudged = _parameter_gradient(problem, phases.positive.state, protocol.β, parameter_ad)
    free = _parameter_gradient(problem, phases.free.state, false, parameter_ad)
    return _gradient_difference(nudged, free, protocol.β)
end

function ep_gradient(
    problem::EPProblem,
    phases::EPPhases,
    protocol::SymmetricEP,
    parameter_ad,
)
    _check_backend(parameter_ad, "parameter")
    plus = _parameter_gradient(problem, phases.positive.state, protocol.β, parameter_ad)
    minus = _parameter_gradient(problem, phases.negative.state, -protocol.β, parameter_ad)
    return _gradient_difference(plus, minus, 2 * protocol.β)
end

ep_gradient(problem::EPProblem, phases::EPPhases, algorithm::EPAlgorithm) =
    ep_gradient(problem, phases, algorithm.protocol, algorithm.parameter_ad)

function ep_gradient(problem::EPProblem, algorithm::EPAlgorithm)
    phases = solve_phases(problem, algorithm)
    gradients = ep_gradient(problem, phases, algorithm)
    return gradients, _stats(problem, phases, gradients)
end
