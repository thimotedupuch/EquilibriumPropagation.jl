abstract type AbstractEPProtocol end
abstract type AbstractEquilibriumAlgorithm end
abstract type AbstractPhase end

"""The unnudged equilibrium phase, corresponding to `β = 0`."""
struct FreePhase <: AbstractPhase end

"""A supervised equilibrium phase with nudging strength `β`."""
struct NudgedPhase{T} <: AbstractPhase
    β::T
end

"""
    OneSidedEP(β)

Estimate the EP gradient from a free phase and one phase nudged by nonzero `β`.
"""
struct OneSidedEP{T} <: AbstractEPProtocol
    β::T
    function OneSidedEP(β::T) where {T}
        iszero(β) && throw(ArgumentError("β must be nonzero"))
        new{T}(β)
    end
end

"""
    SymmetricEP(β)

Estimate the EP gradient by contrasting phases nudged by `+β` and `-β`. This removes
the leading finite-`β` bias and is the recommended default when both phases are stable.
"""
struct SymmetricEP{T} <: AbstractEPProtocol
    β::T
    function SymmetricEP(β::T) where {T}
        iszero(β) && throw(ArgumentError("β must be nonzero"))
        new{T}(β)
    end
end

"""
    Relaxation(; dt=0.1, maxiters=100, abstol=1e-6, reltol=0)

Fixed-step explicit gradient-flow solver for the augmented energy. Iteration stops
when the state-gradient norm is no greater than
`abstol + reltol * initial_residual`, or after `maxiters` updates.
"""
struct Relaxation{T} <: AbstractEquilibriumAlgorithm
    dt::T
    maxiters::Int
    abstol::T
    reltol::T
    function Relaxation(dt::T, maxiters::Integer, abstol::T, reltol::T) where {T}
        dt > zero(T) || throw(ArgumentError("dt must be positive"))
        maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
        abstol >= zero(T) || throw(ArgumentError("abstol must be nonnegative"))
        reltol >= zero(T) || throw(ArgumentError("reltol must be nonnegative"))
        new{T}(dt, Int(maxiters), abstol, reltol)
    end
end

function Relaxation(; dt=0.1, maxiters=100, abstol=1e-6, reltol=zero(abstol))
    T = promote_type(typeof(dt), typeof(abstol), typeof(reltol))
    return Relaxation(convert(T, dt), maxiters, convert(T, abstol), convert(T, reltol))
end

"""
    EPAlgorithm(protocol, solver; state_ad, parameter_ad,
                free_solver=solver, positive_solver=solver, negative_solver=solver)

Bundle an EP protocol, equilibrium solvers, and independent AD backend selectors.
The three phase-specific solvers default to `solver`.
"""
struct EPAlgorithm{P,S,FS,PS,NS,SA,PA}
    protocol::P
    solver::S
    free_solver::FS
    positive_solver::PS
    negative_solver::NS
    state_ad::SA
    parameter_ad::PA
end

function EPAlgorithm(
    protocol,
    solver;
    free_solver=solver,
    positive_solver=solver,
    negative_solver=solver,
    state_ad,
    parameter_ad,
)
    return EPAlgorithm(
        protocol,
        solver,
        free_solver,
        positive_solver,
        negative_solver,
        state_ad,
        parameter_ad,
    )
end

"""
Result of one equilibrium solve.

Fields contain the terminal `state`, state-gradient `residual`, number of
`iterations`, `converged` flag, augmented `energy`, and solved `phase`.
"""
struct EquilibriumSolution{S,T,P}
    state::S
    residual::T
    iterations::Int
    converged::Bool
    energy::T
    phase::P
end

"""
The solutions produced by [`solve_phases`](@ref).

`negative` is `nothing` for [`OneSidedEP`](@ref) and an equilibrium solution for
[`SymmetricEP`](@ref).
"""
struct EPPhases{F,P,N}
    free::F
    positive::P
    negative::N
end

"""
Diagnostics for a completed EP gradient evaluation.

Includes free/positive/negative residuals, iteration counts and energies, supervised
`loss`, `gradient_norm`, phase displacement, and an aggregate `converged` flag.
Negative-phase fields are `missing` for one-sided EP.
"""
struct EPStats{L,FR,PR,NR,FI,PI,NI,EF,EP,EN,G,D}
    loss::L
    free_residual::FR
    positive_residual::PR
    negative_residual::NR
    free_iterations::FI
    positive_iterations::PI
    negative_iterations::NI
    energy_free::EF
    energy_positive::EP
    energy_negative::EN
    gradient_norm::G
    phase_displacement::D
    converged::Bool
end
