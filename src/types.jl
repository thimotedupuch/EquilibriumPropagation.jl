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
    HolomorphicEP(radius; points=8)

Estimate an EP gradient from the first Fourier coefficient of equilibria sampled at
`points` complex nudges on the circle `βₖ = radius * exp(2πim*k/points)`. Unlike
finite-difference EP, the estimator converges to the exact gradient at finite radius
as the number of points increases, provided the augmented energy and equilibrium
branch are holomorphic.

`radius` must be a positive real number and `points` must be at least two. Models must
avoid non-holomorphic operations such as `abs`, `abs2`, `conj`, and adjoints on active
state values.
"""
struct HolomorphicEP{T} <: AbstractEPProtocol
    radius::T
    points::Int
    function HolomorphicEP(radius::T, points::Integer) where {T<:Real}
        radius > zero(radius) || throw(ArgumentError("radius must be positive"))
        points >= 2 || throw(ArgumentError("points must be at least two"))
        new{T}(radius, Int(points))
    end
end


HolomorphicEP(radius::Real; points=8) = HolomorphicEP(radius, points)

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

function _check_tolerances(abstol, reltol)
    abstol >= zero(abstol) || throw(ArgumentError("abstol must be nonnegative"))
    reltol >= zero(reltol) || throw(ArgumentError("reltol must be nonnegative"))
    return nothing
end

"""
    ODERelaxation(algorithm; tspan=(0.0, 100.0), abstol=1e-6, reltol=1e-4,
                  kwargs...)

Optional adaptive ODE integration for `ds/dt = -∂E/∂s`. `algorithm` is an explicit
OrdinaryDiffEq algorithm such as `Tsit5()` or `Rodas5P()`. Loading the matching
OrdinaryDiffEq package activates solving; extra keywords are forwarded to `solve`.
Only array-valued dynamical states are supported by the initial integration.
"""
struct ODERelaxation{A,TS,T,K} <: AbstractEquilibriumAlgorithm
    algorithm::A
    tspan::TS
    abstol::T
    reltol::T
    kwargs::K
end

function ODERelaxation(
    algorithm;
    tspan=(0.0, 100.0),
    abstol=1e-6,
    reltol=1e-4,
    kwargs...,
)
    tspan isa Tuple && length(tspan) == 2 || throw(ArgumentError(
        "tspan must be a tuple containing two endpoints",
    ))
    tspan[2] > tspan[1] || throw(ArgumentError("tspan must have increasing endpoints"))
    T = promote_type(typeof(abstol), typeof(reltol))
    absolute = convert(T, abstol)
    relative = convert(T, reltol)
    _check_tolerances(absolute, relative)
    return ODERelaxation(algorithm, tspan, absolute, relative, (; kwargs...))
end

"""
    SteadyStateRelaxation(algorithm; tspan=Inf, abstol=1e-7, reltol=1e-5,
                          kwargs...)

Optional dynamic steady-state solve using `SteadyStateDiffEq.DynamicSS(algorithm)`.
The supplied algorithm is the ODE integrator used to approach equilibrium. Loading
SteadyStateDiffEq and the algorithm's OrdinaryDiffEq package activates solving.
"""
struct SteadyStateRelaxation{A,TS,T,K} <: AbstractEquilibriumAlgorithm
    algorithm::A
    tspan::TS
    abstol::T
    reltol::T
    kwargs::K
end

function SteadyStateRelaxation(
    algorithm;
    tspan=Inf,
    abstol=1e-7,
    reltol=1e-5,
    kwargs...,
)
    valid_tspan = tspan isa Number ?
        tspan > zero(tspan) :
        tspan isa Tuple && length(tspan) == 2 && tspan[2] > tspan[1]
    valid_tspan || throw(ArgumentError(
        "tspan must be positive or have increasing endpoints",
    ))
    T = promote_type(typeof(abstol), typeof(reltol))
    absolute = convert(T, abstol)
    relative = convert(T, reltol)
    _check_tolerances(absolute, relative)
    return SteadyStateRelaxation(algorithm, tspan, absolute, relative, (; kwargs...))
end

"""
    RootRelaxation(algorithm; abstol=1e-7, reltol=1e-5, kwargs...)

Optional direct solution of `∂E/∂s = 0` using an explicit NonlinearSolve algorithm,
such as `TrustRegion()`. Loading NonlinearSolve activates solving. Extra keywords are
forwarded to `solve`; only array-valued states are initially supported.
"""
struct RootRelaxation{A,T,K} <: AbstractEquilibriumAlgorithm
    algorithm::A
    abstol::T
    reltol::T
    kwargs::K
end

_constant_nudging_schedule(iteration, β) = β
_constant_learning_schedule(iteration) = 1

"""
    ContinuousEP(β, nudged_solver; state_ad, parameter_ad, kwargs...)

Continual-update equilibrium propagation. A free equilibrium is solved first. During
the nudged phase, one explicit state step is alternated with one parameter update
constructed from the parameter-gradient difference between consecutive states.

`nudged_solver` must be a [`Relaxation`](@ref), whose `dt`, `maxiters`, `abstol`, and
`reltol` control the coupled phase. `free_solver` defaults to `nudged_solver` and may
be any equilibrium algorithm. `nudging_schedule(iteration, β)` and
`learning_rate_schedule(iteration)` respectively select the nonzero nudging strength
and multiply the local parameter gradient at each step. Parameter motion has separate
absolute and relative stopping tolerances.
"""
struct ContinuousEP{B,S,FS,SA,PA,NS,LS,T}
    β::B
    nudged_solver::S
    free_solver::FS
    state_ad::SA
    parameter_ad::PA
    nudging_schedule::NS
    learning_rate_schedule::LS
    parameter_abstol::T
    parameter_reltol::T
end

function ContinuousEP(
    β,
    nudged_solver::Relaxation;
    free_solver=nudged_solver,
    state_ad,
    parameter_ad,
    nudging_schedule=_constant_nudging_schedule,
    learning_rate_schedule=_constant_learning_schedule,
    parameter_abstol=nudged_solver.abstol,
    parameter_reltol=nudged_solver.reltol,
)
    iszero(β) && throw(ArgumentError("β must be nonzero"))
    T = promote_type(typeof(parameter_abstol), typeof(parameter_reltol))
    absolute = convert(T, parameter_abstol)
    relative = convert(T, parameter_reltol)
    _check_tolerances(absolute, relative)
    return ContinuousEP(
        β,
        nudged_solver,
        free_solver,
        state_ad,
        parameter_ad,
        nudging_schedule,
        learning_rate_schedule,
        absolute,
        relative,
    )
end

function RootRelaxation(algorithm; abstol=1e-7, reltol=1e-5, kwargs...)
    T = promote_type(typeof(abstol), typeof(reltol))
    absolute = convert(T, abstol)
    relative = convert(T, reltol)
    _check_tolerances(absolute, relative)
    return RootRelaxation(algorithm, absolute, relative, (; kwargs...))
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
`iterations`, `converged` flag, augmented `energy`, solved `phase`, and optional
solver-specific `detail`. Use [`solver_details`](@ref) to access the latter.
"""
struct EquilibriumSolution{S,R,E,P,D}
    state::S
    residual::R
    iterations::Int
    converged::Bool
    energy::E
    phase::P
    detail::D
end

EquilibriumSolution(state, residual, iterations, converged, energy, phase) =
    EquilibriumSolution(state, residual, iterations, converged, energy, phase, nothing)

"""Return optional solver-specific information retained by an equilibrium solution."""
solver_details(solution::EquilibriumSolution) = solution.detail

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

"""Free solution, complex nudges, and sampled solutions from Holomorphic EP."""
struct HolomorphicPhases{F,B,S}
    free::F
    nudges::B
    solutions::S
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

"""
Diagnostics for one [`continuous_ep`](@ref) or [`continuous_train_step!`](@ref).

The `free` and `nudged` equilibrium solutions retain the two endpoint states.
`residual_history` and `parameter_motion_history` expose the coupled dynamics at every
nudged step. `state_converged` and `parameters_stable` report the two independent
stopping tests used to form `converged`.
"""
struct ContinuousEPStats{F,N,L,R,M,D,B,RH,MH}
    free::F
    nudged::N
    loss::L
    residual::R
    parameter_motion::M
    parameter_displacement::D
    β::B
    iterations::Int
    state_converged::Bool
    parameters_stable::Bool
    converged::Bool
    residual_history::RH
    parameter_motion_history::MH
end

"""
    AsymEP(β, solver; state_ad, parameter_ad, free_solver=solver,
           positive_solver=solver, negative_solver=solver)

Asymmetric equilibrium propagation for a [`DynamicalModel`](@ref). Inference follows
the original vector field; opposite nudged phases add its antisymmetric-Jacobian
correction. The estimate approaches the exact equilibrium cost gradient as `β → 0`.
The initial implementation supports array states and [`Relaxation`](@ref).
"""
struct AsymEP{B,S,FS,PS,NS,SA,PA}
    β::B
    solver::S
    free_solver::FS
    positive_solver::PS
    negative_solver::NS
    state_ad::SA
    parameter_ad::PA
end

function AsymEP(β, solver::Relaxation; free_solver=solver, positive_solver=solver,
                negative_solver=solver, state_ad, parameter_ad)
    iszero(β) && throw(ArgumentError("β must be nonzero"))
    return AsymEP(β, solver, free_solver, positive_solver, negative_solver,
                  state_ad, parameter_ad)
end

"""
    DyadicEP(β, solver; state_ad, parameter_ad, free_solver=solver,
             nudged_solver=solver)

Dyadic equilibrium propagation for a [`DynamicalModel`](@ref). Its doubled-state
learning phase preserves the inference dynamics in the midpoint and relaxes the
difference coordinate to an adjoint signal. The estimate approaches the exact
equilibrium cost gradient as `β → 0`. The initial implementation supports array
states and [`Relaxation`](@ref).
"""
struct DyadicEP{B,S,FS,NS,SA,PA}
    β::B
    solver::S
    free_solver::FS
    nudged_solver::NS
    state_ad::SA
    parameter_ad::PA
end

function DyadicEP(β, solver::Relaxation; free_solver=solver, nudged_solver=solver,
                  state_ad, parameter_ad)
    iszero(β) && throw(ArgumentError("β must be nonzero"))
    return DyadicEP(β, solver, free_solver, nudged_solver, state_ad, parameter_ad)
end

"""Diagnostics returned by non-conservative `ep_gradient` evaluations."""
struct NonConservativeStats{M,F,P,N,L,G,A,D}
    method::M
    free::F
    positive::P
    negative::N
    loss::L
    gradient_norm::G
    jacobian_asymmetry::A
    phase_displacement::D
    converged::Bool
end

"""Diagnostics returned by a Holomorphic EP gradient evaluation."""
struct HolomorphicEPStats{P,L,G,I,D}
    phases::P
    loss::L
    gradient_norm::G
    imaginary_leakage::I
    phase_displacement::D
    converged::Bool
end

"""
    ReactantEP(protocol; method=:euler, dt=0.1, free_steps=100, nudged_steps=50,
               abstol=0, reltol=0, damping=1e-4, step_scale=1,
               learning_rate=nothing)

Static-shape equilibrium-propagation algorithm intended for OpenXLA compilation by
the optional Reactant extension. `method` may be `:euler`, `:rk4`, or `:newton`; all
three lower the complete bounded solve to OpenXLA. Newton uses a dense damped Hessian
and currently requires an array-valued state. Nonzero tolerances enable device-side
early stopping: the compiled program retains its static control flow, but a converged
state is no longer updated.
`protocol` may be [`OneSidedEP`](@ref), [`SymmetricEP`](@ref), or
[`HolomorphicEP`](@ref).

When `learning_rate` is provided, the compiled kernel additionally returns one
plain-SGD parameter update. The phase solve and EnzymeMLIR gradients remain the main
accelerated workload.
"""
struct ReactantEP{P,M,T,A,R,D,S,L}
    protocol::P
    method::M
    dt::T
    free_steps::Int
    nudged_steps::Int
    abstol::A
    reltol::R
    damping::D
    step_scale::S
    learning_rate::L
end

function ReactantEP(
    protocol::Union{OneSidedEP,SymmetricEP,HolomorphicEP};
    method=:euler,
    dt=0.1,
    free_steps=100,
    nudged_steps=50,
    abstol=zero(dt),
    reltol=zero(dt),
    damping=convert(typeof(dt), 1e-4),
    step_scale=one(dt),
    learning_rate=nothing,
)
    method in (:euler, :rk4, :newton) || throw(ArgumentError(
        "method must be :euler, :rk4, or :newton",
    ))
    dt > zero(dt) || throw(ArgumentError("dt must be positive"))
    free_steps >= 0 || throw(ArgumentError("free_steps must be nonnegative"))
    nudged_steps >= 0 || throw(ArgumentError("nudged_steps must be nonnegative"))
    _check_tolerances(abstol, reltol)
    damping >= zero(damping) || throw(ArgumentError("damping must be nonnegative"))
    step_scale > zero(step_scale) || throw(ArgumentError("step_scale must be positive"))
    if learning_rate !== nothing
        learning_rate >= zero(learning_rate) || throw(ArgumentError(
            "learning_rate must be nonnegative",
        ))
    end
    return ReactantEP(
        protocol, method, dt, Int(free_steps), Int(nudged_steps), abstol, reltol,
        damping, step_scale, learning_rate,
    )
end
