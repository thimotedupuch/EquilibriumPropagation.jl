module EquilibriumPropagation

import CommonSolve
import ADTypes
using DifferentiationInterface: check_available, gradient, prepare_gradient
using Functors: fcollect, fmap

include("types.jl")
include("model_interface.jl")
include("show.jl")
include("objectives.jl")
include("differentiation.jl")
include("relaxation.jl")
include("phases.jl")
include("gradients.jl")
include("diagnostics.jl")

export EPModel, EPProblem, EPAlgorithm
export FreePhase, NudgedPhase, OneSidedEP, SymmetricEP
export Relaxation, EquilibriumSolution, EPPhases, EPStats
export energy, cost, readout, initial_state, augmented_energy
export equilibrate, solve_phases, ep_gradient, predict
export apply_gradient!, train_step!

"""
    apply_gradient!(optimizer_state, parameters, gradients)

Apply `gradients` with an optimizer and return the updated optimizer state and
parameters. This method is available after loading Optimisers.jl.
"""
function apply_gradient! end

"""
    train_step!(optimizer_state, problem, algorithm)
    train_step!(optimizer_state, parameters, model, (input, target), algorithm;
                model_state=NamedTuple())

Solve the EP phases, compute the parameter gradient, apply one Optimisers.jl update,
and return `(optimizer_state, parameters, stats)`. Loading Optimisers.jl activates
these methods.
"""
function train_step! end

end
