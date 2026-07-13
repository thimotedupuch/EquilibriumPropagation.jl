_missing_residual(solution) = solution === nothing ? missing : solution.residual
_missing_iterations(solution) = solution === nothing ? missing : solution.iterations
_missing_energy(solution) = solution === nothing ? missing : solution.energy

function _tree_norm(tree)
    total = sum(fleaves(tree); init=0) do leaf
        if leaf isa Tuple{}
            return 0
        end
        return leaf isa Number ? abs2(leaf) : sum(abs2, leaf)
    end
    return sqrt(total)
end

function _tree_distance(a, b)
    return _tree_norm(fmap((x, y) -> x .- y, a, b))
end

function _phase_displacement(phases)
    if phases.negative === nothing
        return _tree_distance(phases.positive.state, phases.free.state)
    end
    return (
        positive=_tree_distance(phases.positive.state, phases.free.state),
        negative=_tree_distance(phases.negative.state, phases.free.state),
    )
end

function _stats(problem, phases, gradients)
    negative = phases.negative
    converged = phases.free.converged && phases.positive.converged &&
                (negative === nothing || negative.converged)
    loss = cost(
        problem.model,
        phases.free.state,
        problem.parameters,
        problem.model_state,
        problem.target,
    )
    return EPStats(
        loss,
        phases.free.residual,
        phases.positive.residual,
        _missing_residual(negative),
        phases.free.iterations,
        phases.positive.iterations,
        _missing_iterations(negative),
        phases.free.energy,
        phases.positive.energy,
        _missing_energy(negative),
        _tree_norm(gradients),
        _phase_displacement(phases),
        converged,
    )
end
