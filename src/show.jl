# Compact representations are used when values are nested in another object. The
# richer text/plain methods below are what the REPL and `display` use at top level.
_show_compact(io::IO, value) = show(IOContext(io, :compact => true, :limit => true), value)

Base.show(io::IO, ::FreePhase) = print(io, "FreePhase()")

function Base.show(io::IO, phase::NudgedPhase)
    print(io, "NudgedPhase(")
    _show_compact(io, phase.β)
    return print(io, ')')
end

function Base.show(io::IO, protocol::OneSidedEP)
    print(io, "OneSidedEP(")
    _show_compact(io, protocol.β)
    return print(io, ')')
end

function Base.show(io::IO, protocol::SymmetricEP)
    print(io, "SymmetricEP(")
    _show_compact(io, protocol.β)
    return print(io, ')')
end

function Base.show(io::IO, protocol::HolomorphicEP)
    print(io, "HolomorphicEP(")
    _show_compact(io, protocol.radius)
    return print(io, "; points=", protocol.points, ')')
end

function Base.show(io::IO, solver::Relaxation)
    print(io, "Relaxation(dt=")
    _show_compact(io, solver.dt)
    print(io, ", maxiters=", solver.maxiters, ", abstol=")
    _show_compact(io, solver.abstol)
    print(io, ", reltol=")
    _show_compact(io, solver.reltol)
    return print(io, ')')
end

function Base.show(io::IO, solver::ODERelaxation)
    print(io, "ODERelaxation(")
    _show_compact(io, solver.algorithm)
    print(io, "; tspan=")
    _show_compact(io, solver.tspan)
    print(io, ", abstol=")
    _show_compact(io, solver.abstol)
    print(io, ", reltol=")
    _show_compact(io, solver.reltol)
    return print(io, ')')
end

function Base.show(io::IO, solver::SteadyStateRelaxation)
    print(io, "SteadyStateRelaxation(")
    _show_compact(io, solver.algorithm)
    print(io, "; tspan=")
    _show_compact(io, solver.tspan)
    print(io, ", abstol=")
    _show_compact(io, solver.abstol)
    print(io, ", reltol=")
    _show_compact(io, solver.reltol)
    return print(io, ')')
end

function Base.show(io::IO, solver::RootRelaxation)
    print(io, "RootRelaxation(")
    _show_compact(io, solver.algorithm)
    print(io, "; abstol=")
    _show_compact(io, solver.abstol)
    print(io, ", reltol=")
    _show_compact(io, solver.reltol)
    return print(io, ')')
end

function Base.show(io::IO, algorithm::ContinuousEP)
    print(io, "ContinuousEP(")
    _show_compact(io, algorithm.β)
    print(io, ", ")
    _show_compact(io, algorithm.nudged_solver)
    print(io, "; parameter_abstol=")
    _show_compact(io, algorithm.parameter_abstol)
    print(io, ", parameter_reltol=")
    _show_compact(io, algorithm.parameter_reltol)
    return print(io, ')')
end

function Base.show(io::IO, algorithm::AsymEP)
    print(io, "AsymEP(")
    _show_compact(io, algorithm.β)
    print(io, ", ")
    _show_compact(io, algorithm.solver)
    return print(io, ')')
end

function Base.show(io::IO, algorithm::DyadicEP)
    print(io, "DyadicEP(")
    _show_compact(io, algorithm.β)
    print(io, ", ")
    _show_compact(io, algorithm.solver)
    return print(io, ')')
end

function Base.show(io::IO, algorithm::ReactantEP)
    print(io, "ReactantEP(")
    _show_compact(io, algorithm.protocol)
    print(io, "; dt=")
    _show_compact(io, algorithm.dt)
    print(io, ", free_steps=", algorithm.free_steps,
          ", nudged_steps=", algorithm.nudged_steps)
    if !iszero(algorithm.abstol) || !iszero(algorithm.reltol)
        print(io, ", abstol=")
        _show_compact(io, algorithm.abstol)
        print(io, ", reltol=")
        _show_compact(io, algorithm.reltol)
    end
    print(io, ", learning_rate=")
    _show_compact(io, algorithm.learning_rate)
    return print(io, ')')
end

function Base.show(io::IO, model::EPModel)
    print(io, "EPModel(energy=")
    _show_compact(io, model.energy)
    print(io, ", cost=")
    _show_compact(io, model.cost)
    print(io, ", readout=")
    _show_compact(io, model.readout)
    print(io, ", initial_state=")
    _show_compact(io, model.initial_state)
    return print(io, ')')
end

function Base.show(io::IO, model::DynamicalModel)
    print(io, "DynamicalModel(dynamics=")
    _show_compact(io, model.dynamics)
    print(io, ", cost=")
    _show_compact(io, model.cost)
    print(io, ", readout=")
    _show_compact(io, model.readout)
    print(io, ", initial_state=")
    _show_compact(io, model.initial_state)
    return print(io, ')')
end

function Base.show(io::IO, problem::EPProblem)
    print(io, "EPProblem(model=")
    _show_compact(io, problem.model)
    print(io, ", parameters=")
    summary(io, problem.parameters)
    print(io, ", input=")
    summary(io, problem.input)
    print(io, ", target=")
    summary(io, problem.target)
    return print(io, ')')
end

function Base.show(io::IO, algorithm::EPAlgorithm)
    print(io, "EPAlgorithm(")
    _show_compact(io, algorithm.protocol)
    print(io, ", ")
    _show_compact(io, algorithm.solver)
    if algorithm.free_solver !== algorithm.solver
        print(io, "; free_solver=")
        _show_compact(io, algorithm.free_solver)
        print(io, ',')
    else
        print(io, ';')
    end
    if algorithm.positive_solver !== algorithm.solver
        print(io, " positive_solver=")
        _show_compact(io, algorithm.positive_solver)
        print(io, ',')
    end
    if algorithm.negative_solver !== algorithm.solver
        print(io, " negative_solver=")
        _show_compact(io, algorithm.negative_solver)
        print(io, ',')
    end
    print(io, " state_ad=")
    _show_compact(io, algorithm.state_ad)
    print(io, ", parameter_ad=")
    _show_compact(io, algorithm.parameter_ad)
    return print(io, ')')
end

function Base.show(io::IO, solution::EquilibriumSolution)
    print(io, "EquilibriumSolution(phase=")
    _show_compact(io, solution.phase)
    print(io, ", converged=", solution.converged,
          ", iterations=", solution.iterations, ", residual=")
    _show_compact(io, solution.residual)
    return print(io, ')')
end

function Base.show(io::IO, phases::EPPhases)
    print(io, "EPPhases(free=")
    _show_compact(io, phases.free)
    print(io, ", positive=")
    _show_compact(io, phases.positive)
    print(io, ", negative=")
    _show_compact(io, phases.negative)
    return print(io, ')')
end

function Base.show(io::IO, stats::EPStats)
    print(io, "EPStats(loss=")
    _show_compact(io, stats.loss)
    print(io, ", gradient_norm=")
    _show_compact(io, stats.gradient_norm)
    return print(io, ", converged=", stats.converged, ')')
end

function Base.show(io::IO, stats::ContinuousEPStats)
    print(io, "ContinuousEPStats(iterations=", stats.iterations, ", residual=")
    _show_compact(io, stats.residual)
    print(io, ", parameter_motion=")
    _show_compact(io, stats.parameter_motion)
    return print(io, ", converged=", stats.converged, ')')
end

function Base.show(io::IO, stats::NonConservativeStats)
    print(io, "NonConservativeStats(method=")
    _show_compact(io, stats.method)
    print(io, ", loss=")
    _show_compact(io, stats.loss)
    print(io, ", gradient_norm=")
    _show_compact(io, stats.gradient_norm)
    return print(io, ", converged=", stats.converged, ')')
end

function Base.show(io::IO, stats::HolomorphicEPStats)
    print(io, "HolomorphicEPStats(loss=")
    _show_compact(io, stats.loss)
    print(io, ", gradient_norm=")
    _show_compact(io, stats.gradient_norm)
    print(io, ", imaginary_leakage=")
    _show_compact(io, stats.imaginary_leakage)
    return print(io, ", converged=", stats.converged, ')')
end

function _show_property(io::IO, label, value)
    print(io, label, ": ")
    _show_compact(io, value)
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", model::EPModel)
    print(io, "EPModel")
    _show_property(io, "\n  energy", model.energy)
    _show_property(io, "\n  cost", model.cost)
    _show_property(io, "\n  readout", model.readout)
    _show_property(io, "\n  initial_state", model.initial_state)
end

function Base.show(io::IO, ::MIME"text/plain", model::DynamicalModel)
    print(io, "DynamicalModel")
    _show_property(io, "\n  dynamics", model.dynamics)
    _show_property(io, "\n  cost", model.cost)
    _show_property(io, "\n  readout", model.readout)
    _show_property(io, "\n  initial_state", model.initial_state)
end

function Base.show(io::IO, ::MIME"text/plain", problem::EPProblem)
    print(io, "EPProblem\n  model: ")
    _show_compact(io, problem.model)
    print(io, "\n  parameters: ")
    summary(io, problem.parameters)
    print(io, "\n  model_state: ")
    summary(io, problem.model_state)
    print(io, "\n  input: ")
    summary(io, problem.input)
    print(io, "\n  target: ")
    summary(io, problem.target)
end

function Base.show(io::IO, ::MIME"text/plain", algorithm::EPAlgorithm)
    print(io, "EPAlgorithm")
    _show_property(io, "\n  protocol", algorithm.protocol)
    _show_property(io, "\n  solver", algorithm.solver)
    _show_property(io, "\n  free solver", algorithm.free_solver)
    _show_property(io, "\n  positive solver", algorithm.positive_solver)
    _show_property(io, "\n  negative solver", algorithm.negative_solver)
    _show_property(io, "\n  state AD", algorithm.state_ad)
    _show_property(io, "\n  parameter AD", algorithm.parameter_ad)
end

function Base.show(io::IO, ::MIME"text/plain", solution::EquilibriumSolution)
    print(io, "EquilibriumSolution")
    _show_property(io, "\n  phase", solution.phase)
    _show_property(io, "\n  converged", solution.converged)
    _show_property(io, "\n  iterations", solution.iterations)
    _show_property(io, "\n  residual", solution.residual)
    _show_property(io, "\n  energy", solution.energy)
    print(io, "\n  state: ")
    summary(io, solution.state)
end

function _show_phase_summary(io::IO, label, solution)
    print(io, "\n  ", label, ": ")
    if solution === nothing
        return print(io, "not run")
    end
    print(io, solution.converged ? "converged" : "not converged",
          " (", solution.iterations, " iterations, residual ")
    _show_compact(io, solution.residual)
    return print(io, ')')
end

function Base.show(io::IO, ::MIME"text/plain", phases::EPPhases)
    print(io, "EPPhases")
    _show_phase_summary(io, "free", phases.free)
    _show_phase_summary(io, "positive", phases.positive)
    _show_phase_summary(io, "negative", phases.negative)
end

function _show_stats_phase(io::IO, label, residual, iterations, energy)
    print(io, "\n    ", label, ": residual=")
    _show_compact(io, residual)
    print(io, ", iterations=")
    _show_compact(io, iterations)
    print(io, ", energy=")
    _show_compact(io, energy)
end

function Base.show(io::IO, ::MIME"text/plain", stats::EPStats)
    print(io, "EPStats")
    _show_property(io, "\n  converged", stats.converged)
    _show_property(io, "\n  loss", stats.loss)
    _show_property(io, "\n  gradient norm", stats.gradient_norm)
    _show_property(io, "\n  phase displacement", stats.phase_displacement)
    print(io, "\n  phases:")
    _show_stats_phase(io, "free", stats.free_residual, stats.free_iterations,
                      stats.energy_free)
    _show_stats_phase(io, "positive", stats.positive_residual,
                      stats.positive_iterations, stats.energy_positive)
    _show_stats_phase(io, "negative", stats.negative_residual,
                      stats.negative_iterations, stats.energy_negative)
end
