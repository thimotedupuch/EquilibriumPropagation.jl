If we were to create modern julia package EquilibriumPropagation.jl to use Julia ML and SciML world class software stack to create a unified framework to work with EP, what should the package look like ? (dependencies, features, usage) 
It would be the occasion to use techniques not common for codes using pytorch or jax for tensor computation ?


I should specify that it is not really related to DeepEquilibriumNetworks.jl that is almost legacy (using the old julia stack)

And this package should be smaller in scope and less ambitious

Réfléchi pendant 4m 36s
Recommended positioning
EquilibriumPropagation.jl should be an algorithmic package for training energy-based systems through equilibrium propagation, not a new neural-network framework and not a general implicit-layer library.

Its central abstraction would be:

Φ β ( s ; θ , x , y ) = E ( s ; θ , x ) + β   C ( r ( s ) , y ) , Φ  β ​
 (s;θ,x,y)=E(s;θ,x)+βC(r(s),y),
where 
s
s is the dynamical state, 
θ
θ the trainable parameters, 
E
E the internal energy, 
r
r a readout and 
C
C the supervised cost. The package finds stationary states of 
Φ
β
Φ 
β
​
  and constructs an EP parameter gradient from contrasts between phases. This corresponds directly to the original EP formulation. 

This makes the conceptual boundary with DeepEquilibriumNetworks.jl clear. DEQ software represents an implicit neural-network layer and differentiates the output of a fixed-point solve. EquilibriumPropagation.jl would instead implement a learning rule based on multiple physical or numerical equilibria; ordinary EP training should not differentiate through the equilibrium solver. DeepEquilibriumNetworks.jl describes itself as a framework for implicit layers built on DifferentialEquations and Lux, which is a different problem. 

A useful one-sentence scope would be:

Define an energy, find free and nudged equilibria, compute an EP gradient, and update structured Julia parameters.

Core package architecture
I would use four principal abstractions.

EPModel
EPProblem
AbstractEquilibriumSolver
AbstractEPProtocol
A proposed low-level interface could be:

model = EPModel(
    energy     = (s, ps, x, st) -> E,
    cost       = (s, y, ps, st) -> C,
    readout    = (s, ps, st) -> ŷ,
    initialstate = (ps, x, st) -> s₀,
)

prob = EPProblem(model, ps, st, x, y)

sol = equilibrate(prob, FreePhase(), solver; ad = state_ad)

grads, stats = ep_gradient(
    prob,
    SymmetricEP(β);
    solver,
    state_ad,
    parameter_ad,
)
The contract should be deliberately small:

energy returns a scalar.

cost returns a scalar.

initialstate constructs the dynamical state.

readout is mostly for prediction and diagnostics.

Model state st is constant during one equilibrium solve.

The latter restriction matters. Dropout, BatchNorm statistics and other evolving layer state would make the energy landscape change while it is being minimized. A Lux adapter should therefore place such layers in test mode or reject state changes during relaxation.

Protocols
For version 0.1, three protocols are enough:

OneSidedEP(β)
SymmetricEP(β)
RandomSignEP(β, rng)
SymmetricEP should probably be the recommended default:

g θ ≈ ∂ θ Φ + β ( s + β ) − ∂ θ Φ − β ( s − β ) 2 β . g  θ ​ ≈  2β ∂  θ ​ Φ  +β ​ (s  +β ​ )−∂  θ ​ Φ  −β ​ (s  −β ​ )
​
 .
Symmetric positive and negative nudging reduces the leading finite-
β
β bias from first to second order and has been important in practical deep EP experiments. 

An important implementation detail is that the parameter derivatives must treat the equilibrated states as constants. Otherwise AD may accidentally differentiate through the solver, changing EP into an implicit-gradient method.

Solvers
The package should separate the EP protocol from the equilibrium algorithm:

Relaxation(; dt, maxiters, abstol)
FixedPointIteration(; damping, maxiters)
NonlinearSolveJL(alg)
SteadyStateJL(alg)
Relaxation should live in the core package. It would implement something like

s k + 1 = s k − Δ t   ∇ s Φ β ( s k ) , s  k+1 ​ =s  k ​ −Δt∇  s ​ Φ  β ​ (s  k ​ ),
using preallocated state-gradient buffers.

The SciML extensions could translate the stationarity condition

∇ s Φ β ( s ) = 0 ∇  s ​ Φ  β ​ (s)=0
into a NonlinearProblem, or integrate the gradient flow until a steady state. NonlinearSolve offers robust nonlinear polyalgorithms, sparse Jacobians, iterative linear solvers and Jacobian-free Newton–Krylov methods. Steady-state solvers can instead approach equilibria through ODE integration. 

This gives one of the package’s strongest Julia-specific features:

solve(prob, Relaxation(...))
solve(prob, NonlinearSolveJL(NewtonRaphson(...)))
solve(prob, SteadyStateJL(DynamicSS(...)))
The model and EP algorithm remain unchanged.

Dependencies
I would keep the mandatory dependency graph small.

[deps]
ADTypes
CommonSolve
DifferentiationInterface
Optimisers

# plus stdlibs:
LinearAlgebra
Random
Statistics
DifferentiationInterface provides backend-independent gradients, JVPs, VJPs, Hessian-vector products, in-place differentiation and reusable preparation objects. Reusing prepared differentiation state is especially relevant because the same energy derivatives are evaluated repeatedly across batches and phases. 

Optimisers.jl provides updates over nested parameter structures and is already used by Lux. It operates on arrays, named tuples and structures understood by Functors, so the package would not need its own optimizer hierarchy. 

Everything substantial should be an extension or weak dependency:

[weakdeps]
Lux
MLDataDevices
NonlinearSolve
LinearSolve
SteadyStateDiffEq
ComponentArrays
ChainRulesCore
Reactant
Julia package extensions are specifically designed to load integration code only when optional dependencies are present. 

AD implementations such as Enzyme, Mooncake, Zygote or ForwardDiff should be selected explicitly by the user through ADTypes:

state_ad     = AutoEnzyme()
parameter_ad = AutoMooncake()
Lux currently supports several such backends through ADTypes, including Enzyme, Mooncake, Zygote, ReverseDiff and ForwardDiff. 

I would avoid an opaque “automatic best AD backend” policy. Different derivatives have different dimensionality and mutation requirements, so reproducible explicit selection is preferable.

High-level training API
A small convenience layer could resemble Lux and Optimisers without replacing either:

alg = EPAlgorithm(
    protocol = SymmetricEP(0.05f0),
    solver   = Relaxation(
        dt       = 0.2f0,
        maxiters = 100,
        abstol   = 1f-5,
    ),
    state_ad     = AutoEnzyme(),
    parameter_ad = AutoMooncake(),
)

opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps)

opt_state, ps, stats = train_step!(
    opt_state,
    ps,
    model,
    (x, y),
    alg,
)
train_step! would perform:

Free equilibrium from initialstate.

Positive and negative phases warm-started from the free equilibrium.

Parameter-gradient contrast.

Optimisers.update!.

Return structured diagnostics.

It should also expose the pieces independently:

phases = solve_phases(prob, alg)
grads  = ep_gradient(prob, phases, alg)
ŷ      = predict(phases.free, model)
That is important for research code, where users frequently need to modify a phase, inspect states or replace the update rule.

A minimal complete quadratic example might look like:

using EquilibriumPropagation
using ADTypes, Enzyme, Optimisers

energy(s, ps, x, st) =
    sum(abs2, s) / 2 -
    sum(s .* (ps.W * x .+ ps.b))

cost(s, y, ps, st) =
    sum(abs2, s .- y) / 2

model = EPModel(
    energy = energy,
    cost = cost,
    readout = (s, ps, st) -> s,
    initialstate = (ps, x, st) -> zero(ps.W * x),
)

ps = (
    W = 0.01f0 .* randn(Float32, 10, 20),
    b = zeros(Float32, 10, 1),
)
st = NamedTuple()

alg = EPAlgorithm(
    SymmetricEP(0.05f0),
    Relaxation(dt=0.2f0, maxiters=100);
    state_ad=AutoEnzyme(),
    parameter_ad=AutoEnzyme(),
)

opt = Optimisers.setup(Optimisers.Adam(1f-3), ps)
opt, ps, stats = train_step!(opt, ps, model, (x, y), alg)
The quadratic model should actually be one of the package’s analytical tests: its equilibria and exact supervised gradient are available in closed form.

Features worth including
The first release should focus on numerical correctness rather than a model zoo.

Essential features:

Free, positive and negative phases.

Warm starts from the free equilibrium.

Separate tolerances and iteration limits per phase.

Fixed-step, allocation-conscious relaxation.

Pluggable state and parameter AD.

Optimisers.jl integration.

Lux parameter/state adapter.

CPU and generic GPU-array compatibility.

Convergence and gradient diagnostics.

Checkpointable explicit training state.

Analytical and finite-difference test models.

The diagnostics are particularly important because EP has at least three independent approximation errors:

gradient error
=
finite- β error + equilibrium error + numerical differentiation error . gradient error=finite-β error+equilibrium error+numerical differentiation error.
A useful EPStats object should contain:

stats.loss
stats.free_residual
stats.positive_residual
stats.negative_residual
stats.free_iterations
stats.positive_iterations
stats.negative_iterations
stats.energy_free
stats.energy_positive
stats.energy_negative
stats.gradient_norm
stats.phase_displacement
stats.converged
Tests should verify that the EP gradient approaches an exact implicit or finite-difference gradient as both 
β
β and equilibrium residuals decrease.

Julia techniques that would be genuinely valuable
Yes—this is an unusually good opportunity to exploit methods that are uncommon in typical PyTorch or JAX EP implementations. These methods are not impossible there, but Julia can make them part of the ordinary package architecture rather than specialized external machinery.

1. Matrix-free Newton–Krylov equilibria
The equilibrium residual is

R
(
s
)
=
∇
s
Φ
β
(
s
)
.
R(s)=∇ 
s
​
 Φ 
β
​
 (s).
Newton’s method requires solving

∇
s
2
Φ
β
(
s
)
 
δ
s
=
−
R
(
s
)
.
∇ 
s
2
​
 Φ 
β
​
 (s)δs=−R(s).
There is no need to construct the Hessian explicitly. DifferentiationInterface can supply Hessian-vector products, while NonlinearSolve and LinearSolve can use matrix-free Krylov methods and preconditioners. NonlinearSolve automatically enters Jacobian-free mode with appropriate Krylov linear solvers. 

For large EP states, this may be substantially better than running thousands of explicit Euler relaxation steps.

2. Different AD modes for different operations
The package can independently choose algorithms for:

∇
s
Φ
∇ 
s
​
 Φ,

∇
θ
Φ
∇ 
θ
​
 Φ,

Hessian-vector products in state space,

Jacobians used only for diagnostics.

For example:

state_gradient_ad = AutoEnzyme()
parameter_ad      = AutoMooncake()
hvp_ad            = AutoForwardFromReverse(...)
DifferentiationInterface’s preparation API can cache tapes, buffers or backend-specific setup across training steps. 

3. Mutating relaxation kernels
A performant implementation could provide both:

relax(...)
relax!(cache, ...)
The in-place version can reuse state, gradient and residual buffers. Enzyme operates on compiler IR and supports differentiation patterns involving mutated arrays and custom rules, making allocation-conscious simulation kernels a realistic design target. 

EP is especially suitable because ordinary training only needs terminal states, not a tape containing every relaxation step.

4. Sparse energy systems
Physical networks, lattice models and locally connected EP systems often produce sparse state Jacobians or Hessians. DifferentiationInterface and the SciML differentiation stack support sparse differentiation and reusable sparsity/coloring preparation. 

A user could provide:

jac_prototype
sparsity_detector
preconditioner
without changing the EP protocol.

5. Local energy terms with analytic updates
A useful optional abstraction would be:

EnergySum(
    QuadraticPotential(...),
    BilinearCoupling(...),
    BiasPotential(...),
)
The generic fallback would use AD. Built-in energy terms could instead implement analytical local parameter derivatives through multiple dispatch.

For example, a bilinear coupling can produce its parameter update directly from pre- and post-synaptic equilibrium states. This preserves the local-learning interpretation of EP and avoids running reverse-mode AD over an entire model merely to recover a simple outer product.

This should remain a small collection of primitives, not grow into a symbolic modeling language.

6. Continuation and parallel phases
The positive and negative phases can be:

run concurrently from the same free state;

reached by continuation in 
β
β;

solved using cached nonlinear-solver state;

assigned different convergence tolerances.

Continuation and warm-starting are standard numerical-analysis ideas but are less common in tensor-centric EP implementations. NonlinearSolve also provides continuation machinery and flexible solver composition. 

7. Structured heterogeneous states
The dynamical state need not be one dense tensor. It could be a NamedTuple, ComponentArray, static array, sparse array or a problem-specific array type. ComponentArrays and SciMLStructures provide mechanisms for exposing structured data to ordinary vector-based numerical algorithms while retaining named components. 

This would make the package useful beyond layered neural networks—for coupled fields, circuits, mechanical networks and hybrid physical models.

8. Compiled fixed-iteration accelerator kernels
An optional Reactant extension could compile a fixed-number relaxation loop to MLIR/XLA for CPUs, GPUs or TPUs. Reactant currently compiles Julia functions through MLIR and XLA, but assumes traced control-flow patterns, so fixed-iteration or masked-convergence relaxation is a more natural target than a highly dynamic nonlinear polyalgorithm. 

What not to put in version 0.1
To preserve the smaller scope, I would explicitly exclude:

A general DEQ or implicit-layer API.

Differentiation through arbitrary equilibrium solvers as the primary training method.

Spiking-neuron simulation.

Neuromorphic hardware backends.

Dataset and data-loader abstractions.

Distributed training infrastructure.

A large collection of network architectures.

General non-conservative dynamics.

Holomorphic and continual EP as stable APIs.

Holomorphic EP is scientifically interesting because it can recover exact gradients from finite-amplitude complex oscillations, but it introduces complex dynamics, Fourier extraction and complex-AD semantics. 

Continual EP updates weights during relaxation and changes the relationship between solver state and optimizer state. It deserves a separate protocol implementation after conventional EP is robust. 

Likewise, recent non-conservative EP formulations introduce Jacobian antisymmetry corrections or augmented state spaces. They are promising research extensions, but including them initially would substantially expand the mathematical and testing surface. 

A realistic release sequence
Version 0.1: function-based conservative energies, Relaxation, one-sided and symmetric EP, DifferentiationInterface, Optimisers, diagnostics and analytical gradient tests.

Version 0.2: Lux and MLDataDevices extensions, batched execution, preallocated caches and accelerator-compatible fixed-step loops.

Version 0.3: NonlinearSolve and SteadyStateDiffEq extensions, matrix-free Newton–Krylov solving, sparsity and preconditioning.

Experimental thereafter: local energy primitives, Reactant compilation, continual EP and holomorphic EP.

The package’s strongest identity would therefore not be “more EP architectures.” It would be:

A compact, solver-agnostic and AD-agnostic implementation of equilibrium propagation, with unusually strong numerical methods and explicit control over every approximation involved.


Should we use ADTypes or DifferentiationInterface ? Give the filetree of the project 
Use both
EquilibriumPropagation.jl should depend directly on both packages, with a strict separation of responsibilities:

ADTypes.jl describes the selected backend.

DifferentiationInterface.jl performs differentiation.

ADTypes provides configuration objects such as AutoForwardDiff(), AutoEnzyme(), AutoMooncake() and AutoZygote(). It does not provide the common gradient, hvp or preparation machinery required by the package. DifferentiationInterface consumes those backend objects and provides backend-independent operators such as gradient!, value_and_gradient, prepare_gradient, hvp! and sparse differentiation. 

Conceptually:

using ADTypes: AutoForwardDiff, AutoMooncake
using DifferentiationInterface: gradient, gradient!, prepare_gradient

alg = EPAlgorithm(
    protocol = SymmetricEP(0.05f0),
    solver = Relaxation(dt=0.1f0, maxiters=100),
    state_ad = AutoForwardDiff(),
    parameter_ad = AutoMooncake(),
)
Internally:

gradient(state_objective, alg.state_ad, state)
gradient(parameter_objective, alg.parameter_ad, parameters)
Do not create another AD abstraction
The package should not define something like:

abstract type AbstractEPADBackend end
That would duplicate ADTypes. Store the ADTypes-compatible backend object directly:

struct EPAlgorithm{P,S,StateAD,ParameterAD}
    protocol::P
    solver::S
    state_ad::StateAD
    parameter_ad::ParameterAD
end
The fields should probably remain unconstrained rather than explicitly requiring <: AbstractADType, because DifferentiationInterface also provides composite backend wrappers such as SecondOrder.

Backend availability should be validated when constructing a cache or starting a solve:

function check_backends(alg)
    DifferentiationInterface.check_available(alg.state_ad) ||
        throw(ArgumentError("The state AD backend is not loaded"))

    DifferentiationInterface.check_available(alg.parameter_ad) ||
        throw(ArgumentError("The parameter AD backend is not loaded"))

    return nothing
end
DifferentiationInterface.check_available specifically checks whether the extension for a backend has been loaded. 

Backend packages should not be package dependencies
ForwardDiff, Enzyme, Mooncake and Zygote should not be mandatory dependencies or weak dependencies of EquilibriumPropagation.jl.

The user environment would contain the desired implementation:

using EquilibriumPropagation
using ForwardDiff
using Mooncake
using ADTypes

alg = EPAlgorithm(
    SymmetricEP(0.05),
    Relaxation();
    state_ad = AutoForwardDiff(),
    parameter_ad = AutoMooncake(),
)
DifferentiationInterface itself uses package extensions to activate the corresponding backend implementation when its package is loaded. This avoids reproducing its extension system inside EquilibriumPropagation.jl. 

The backend implementation packages belong in:

documentation environments;

examples;

test environments;

user projects.

They do not belong in the core [deps].

Core dependency set
I recommend the following mandatory dependencies.

Dependency	Purpose
ADTypes	Backend configuration objects in the public API
DifferentiationInterface	Gradients, in-place gradients, preparation and HVPs
CommonSolve	Standard solve, init, step! and solve! names
Functors	Operations over structured parameter and gradient trees
LinearAlgebra	Norms and residual calculations
Random	Random-sign protocols and initialization
Statistics	Optional batch diagnostics
CommonSolve is small and allows EP solvers to expose the usual Julia scientific-computing interface without defining a conflicting private solve function. 

Functors.fmap is useful for combining gradients stored as nested named tuples or model structures:

ep_gradient = fmap(
    (gplus, gminus) -> (gplus - gminus) / (2β),
    gradient_plus,
    gradient_minus,
)
Functors provides structural traversal and reconstruction independently of Lux or Optimisers. 

Recommended weak dependencies:

Lux
Optimisers
NonlinearSolve
Possible later weak dependencies:

ComponentArrays
SteadyStateDiffEq
Reactant
SparseMatrixColorings
SparseConnectivityTracer
Those should not be present in the first release unless their integration is implemented and tested.

Recommended project tree
For a smaller-scope package, I would keep src/ mostly flat. Deep directory hierarchies would be premature.

EquilibriumPropagation.jl/
├── Project.toml
├── README.md
├── LICENSE
├── .JuliaFormatter.toml
│
├── src/
│   ├── EquilibriumPropagation.jl
│   ├── types.jl
│   ├── model_interface.jl
│   ├── objectives.jl
│   ├── differentiation.jl
│   ├── caches.jl
│   ├── solver_interface.jl
│   ├── relaxation.jl
│   ├── phases.jl
│   ├── gradients.jl
│   └── diagnostics.jl
│
├── ext/
│   ├── EquilibriumPropagationLuxExt.jl
│   ├── EquilibriumPropagationOptimisersExt.jl
│   └── EquilibriumPropagationNonlinearSolveExt.jl
│
├── test/
│   ├── Project.toml
│   ├── runtests.jl
│   ├── test_model_interface.jl
│   ├── test_objectives.jl
│   ├── test_relaxation.jl
│   ├── test_phases.jl
│   ├── test_gradients.jl
│   ├── test_diagnostics.jl
│   ├── test_allocations.jl
│   │
│   └── extensions/
│       ├── test_lux.jl
│       ├── test_optimisers.jl
│       └── test_nonlinear_solve.jl
│
├── docs/
│   ├── Project.toml
│   ├── make.jl
│   └── src/
│       ├── index.md
│       ├── getting_started.md
│       ├── model_interface.md
│       ├── differentiation.md
│       ├── equilibrium_solvers.md
│       ├── lux_integration.md
│       └── api.md
│
├── examples/
│   ├── Project.toml
│   ├── quadratic_system.jl
│   ├── layered_energy_network.jl
│   └── lux_energy_model.jl
│
├── benchmark/
│   ├── Project.toml
│   ├── benchmarks.jl
│   ├── ad_backends.jl
│   └── relaxation.jl
│
Julia extensions are automatically loaded when their trigger packages are loaded, allowing the Lux and SciML integrations to remain outside the core package. 

Responsibilities of the source files
EquilibriumPropagation.jl
Only module definition, imports, exports and includes:

module EquilibriumPropagation

using ADTypes
using CommonSolve
using DifferentiationInterface
using Functors
using LinearAlgebra
using Random
using Statistics

include("types.jl")
include("model_interface.jl")
include("objectives.jl")
include("differentiation.jl")
include("caches.jl")
include("solver_interface.jl")
include("relaxation.jl")
include("phases.jl")
include("gradients.jl")
include("diagnostics.jl")

export EPModel, EPProblem
export FreePhase, NudgedPhase
export OneSidedEP, SymmetricEP
export Relaxation
export equilibrate, solve_phases, ep_gradient
export EPStats

end
types.jl
Only central data structures and abstract interfaces:

abstract type AbstractEPProtocol end
abstract type AbstractEquilibriumAlgorithm end
abstract type AbstractPhase end

struct FreePhase <: AbstractPhase end

struct NudgedPhase{T} <: AbstractPhase
    β::T
end

struct OneSidedEP{T} <: AbstractEPProtocol
    β::T
end

struct SymmetricEP{T} <: AbstractEPProtocol
    β::T
end

struct EPAlgorithm{P,S,SA,PA}
    protocol::P
    solver::S
    state_ad::SA
    parameter_ad::PA
end
model_interface.jl
Defines the minimal user interface:

energy(model, state, parameters, model_state, input)
cost(model, state, parameters, model_state, target)
readout(model, state, parameters, model_state)
initial_state(model, parameters, model_state, input)
It should also contain the function-based convenience model:

struct EPModel{E,C,R,I}
    energy::E
    cost::C
    readout::R
    initial_state::I
end
objectives.jl
Defines the augmented energy:

augmented_energy(
    model,
    state,
    parameters,
    model_state,
    input,
    target,
    β,
)
This file must contain no AD calls.

differentiation.jl
This should be the only core source file that calls DifferentiationInterface.

It would implement private functions such as:

_state_gradient
_state_gradient!
_parameter_gradient
_prepare_state_gradient
_prepare_parameter_gradient
Keeping all DI calls in one place makes backend problems easier to diagnose and test.

A typical state derivative can use DI contexts:

function state_objective(
    state,
    model,
    parameters,
    model_state,
    input,
    target,
    β,
)
    return augmented_energy(
        model,
        state,
        parameters,
        model_state,
        input,
        target,
        β,
    )
end

function state_gradient(
    backend,
    state,
    model,
    parameters,
    model_state,
    input,
    target,
    β,
)
    return gradient(
        state_objective,
        backend,
        state,
        Constant(model),
        Constant(parameters),
        Constant(model_state),
        Constant(input),
        Constant(target),
        Constant(β),
    )
end
DI contexts allow a function to have one active argument while the other arguments are explicitly treated as constants or mutable caches. 

A closure-based fallback should exist for backends that do not support the required context type.

caches.jl
Contains prepared differentiation and relaxation buffers:

struct RelaxationCache{S,G,P}
    state::S
    gradient::G
    gradient_prep::P
end
Preparation should initially happen once per phase solve. Later, caches can be reused between batches when types and dimensions remain unchanged.

DI preparation objects can preallocate buffers or record backend-specific traces, but they are valid only under documented type, size and function constraints. They are also not thread-safe, so each concurrent solve needs its own cache. 

solver_interface.jl
Defines:

equilibrate(problem, phase, algorithm)
CommonSolve.solve(problem, algorithm)
CommonSolve.init(problem, algorithm)
CommonSolve.step!(cache)
CommonSolve.solve!(cache)
It should contain interfaces only, not the actual relaxation implementation.

relaxation.jl
Contains the built-in gradient-flow solver:

struct Relaxation{T}
    dt::T
    maxiters::Int
    abstol::T
    reltol::T
end
This should remain the only mandatory equilibrium algorithm in version 0.1.

phases.jl
Orchestrates:

solve_free_phase
solve_positive_phase
solve_negative_phase
solve_phases
It handles warm starting but does not compute parameter gradients.

gradients.jl
Implements the EP estimators:

ep_gradient(problem, phases, ::OneSidedEP, backend)
ep_gradient(problem, phases, ::SymmetricEP, backend)
This file combines phase-local parameter derivatives using Functors.fmap.

diagnostics.jl
Defines:

struct EPStats
    loss
    free_residual
    positive_residual
    negative_residual
    free_iterations
    positive_iterations
    negative_iterations
    gradient_norm
    converged
end
Extensions
EquilibriumPropagationLuxExt.jl
This should provide Lux-specific convenience, not alter the core mathematical interface.

Possible responsibilities:

initial_state(::LuxEnergyModel, ps, st, x)
readout(::LuxEnergyModel, s, ps, st)
Lux.testmode integration
Lux parameter-tree validation
The extension CI should explicitly test the supported combinations of Lux parameter structures and DI backends. DifferentiationInterface currently includes Lux-oriented test scenarios, but labels them experimental, so the package should not claim universal compatibility across every backend and every Lux model. 

EquilibriumPropagationOptimisersExt.jl
Provides:

train_step!
apply_gradient!
The core package returns gradients; this extension applies them using Optimisers.

That preserves the distinction:

phases = solve_phases(problem, algorithm)
gradient, stats = ep_gradient(problem, phases, algorithm)
versus the convenience operation:

opt_state, parameters, stats =
    train_step!(opt_state, parameters, problem, algorithm)
EquilibriumPropagationNonlinearSolveExt.jl
Converts the equilibrium condition

∇
s
Φ
β
(
s
)
=
0
∇ 
s
​
 Φ 
β
​
 (s)=0
into a NonlinearProblem and forwards it to NonlinearSolve.solve.

The extension should not expose every SciML type through the main namespace. One wrapper algorithm is sufficient:

NonlinearEquilibrium(solver; kwargs...)
Suggested Project.toml organization
Conceptually:

[deps]
ADTypes = "..."
CommonSolve = "..."
DifferentiationInterface = "..."
Functors = "..."

[weakdeps]
Lux = "..."
NonlinearSolve = "..."
Optimisers = "..."

[extensions]
EquilibriumPropagationLuxExt = "Lux"
EquilibriumPropagationNonlinearSolveExt = "NonlinearSolve"
EquilibriumPropagationOptimisersExt = "Optimisers"

[compat]
julia = "..."
ADTypes = "..."
CommonSolve = "..."
DifferentiationInterface = "..."
Functors = "..."
The exact compat bounds should be established from CI rather than copied blindly from this sketch.

What should not appear in the initial tree
Do not add these initially:

src/backends/enzyme.jl
src/backends/zygote.jl
src/backends/forwarddiff.jl
src/backends/mooncake.jl
src/deq.jl
src/implicit_differentiation.jl
src/spiking/
src/hardware/
src/distributed/
The backend-specific files would duplicate DifferentiationInterface’s job. The other directories would broaden the project beyond its intended scope.

The clean dependency flow should be:

AD backend implementation
        ↓
      ADTypes
        ↓
DifferentiationInterface
        ↓
EquilibriumPropagation differentiation.jl
        ↓
relaxation and EP gradient estimators
That gives the package backend independence without creating another abstraction layer.

﻿
 

