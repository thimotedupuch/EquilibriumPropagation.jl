module EquilibriumPropagation

import CommonSolve
import ADTypes
using DifferentiationInterface: check_available, gradient, jacobian, prepare_gradient
using Functors: fleaves, fmap
using LinearAlgebra: Diagonal, diag, dot

include("types.jl")
include("model_interface.jl")
include("hopfield.jl")
include("show.jl")
include("objectives.jl")
include("differentiation.jl")
include("relaxation.jl")
include("phases.jl")
include("gradients.jl")
include("diagnostics.jl")
include("continuous.jl")
include("nonconservative.jl")
include("holomorphic.jl")

export EPModel, DynamicalModel, EPProblem, EPBatch, batch_size, EPAlgorithm
export FreePhase, NudgedPhase, OneSidedEP, SymmetricEP, HolomorphicEP
export Relaxation, EquilibriumSolution, EPPhases, HolomorphicPhases, EPStats
export ODERelaxation, SteadyStateRelaxation, RootRelaxation, solver_details
export ContinuousEP, ContinuousEPStats, continuous_ep, continuous_train_step!
export AsymEP, DyadicEP, NonConservativeStats
export HolomorphicEPStats
export energy, vector_field, cost, readout, initial_state, augmented_energy
export ContinuousHopfield, QuadraticPotential, HardClamp, BipolarSquaredError
export ZeroState, GlorotUniform, setup, pack_parameters, unpack_parameters
export equilibrate, solve_phases, ep_gradient, predict
export apply_gradient!, train_step!
export lux_setup, lux_energy_model, lux_dynamical_model
export ReactantEP, reactant_inputs, compile_reactant

"""
    reactant_inputs(problem; backend=nothing)

Transfer a conservative array-state problem's parameters, model state, batch, and
initial state to Reactant device arrays. Loading Reactant and Enzyme activates this
function. Select `backend="cpu"`, `"gpu"`, or `"tpu"` before transfer when desired.
"""
function reactant_inputs end

"""
    compile_reactant(problem, algorithm::ReactantEP; backend=nothing, kwargs...)

Compile fixed-step batched EP relaxation and EnzymeMLIR gradient extraction through
Reactant/OpenXLA. Returns a reusable executable that owns its initial device buffers
and also accepts same-shaped device parameters, model state, batch, and initial state.
Loading Reactant and Enzyme activates this function.
"""
function compile_reactant end

"""
    lux_setup(rng, layer) -> parameters, model_state

Initialize a Lux layer and place its non-trainable state in test mode for use during
equilibrium relaxation. Loading Lux.jl activates this function.
"""
function lux_setup end

"""
    lux_energy_model(layer; cost, readout, initial_state, kwargs...)

Adapt a Lux layer that contributes a scalar energy into an [`EPModel`](@ref). Loading
Lux.jl activates this function. The Lux state must remain unchanged during every
relaxation evaluation.
"""
function lux_energy_model end

"""
    lux_dynamical_model(layer; cost, readout, initial_state, kwargs...)

Adapt a Lux layer that computes a vector field into a [`DynamicalModel`](@ref).
Loading Lux.jl activates this function. The Lux state must remain unchanged during
every relaxation evaluation.
"""
function lux_dynamical_model end

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

"""
    continuous_train_step!(optimizer_state, problem, algorithm)

Run [`continuous_ep`](@ref), applying each local parameter gradient through an
Optimisers.jl state. Loading Optimisers.jl activates this method.
"""
function continuous_train_step! end

end
