# Optional equilibrium solvers

The core `Relaxation` algorithm uses fixed-step explicit Euler updates. Three optional
wrappers integrate the same `EPProblem` and phase interfaces with SciML solvers:

- `ODERelaxation` follows the gradient-flow ODE until its residual tolerance is met or
  its time horizon ends;
- `SteadyStateRelaxation` uses `SteadyStateDiffEq.DynamicSS` to integrate until the
  derivative is sufficiently small; and
- `RootRelaxation` solves the state-gradient equation directly with NonlinearSolve.

The wrappers keep SciML packages optional. Load the package that owns an algorithm
before constructing it:

```julia
using OrdinaryDiffEqTsit5

solver = ODERelaxation(
    Tsit5();
    tspan=(0.0, 100.0),
    abstol=1e-6,
    reltol=1e-4,
)
solution = equilibrate(problem, FreePhase(), solver; state_ad=state_ad)
```

For stiff gradient flow, use the Rosenbrock integration:

```julia
using OrdinaryDiffEqRosenbrock

solver = ODERelaxation(
    Rodas5P();
    tspan=(0.0, 100.0),
    abstol=1e-7,
    reltol=1e-5,
)
```

Dynamic steady-state integration accepts an ODE algorithm explicitly:

```julia
using OrdinaryDiffEqRosenbrock, SteadyStateDiffEq

solver = SteadyStateRelaxation(
    Rodas5P();
    abstol=1e-7,
    reltol=1e-5,
)
```

Direct root finding uses a NonlinearSolve algorithm and benefits from the free phase
as the initial guess for each nudged phase through the existing phase warm starts:

```julia
using NonlinearSolve

solver = RootRelaxation(TrustRegion(); abstol=1e-7, reltol=1e-5)
algorithm = EPAlgorithm(
    SymmetricEP(0.05),
    solver;
    state_ad,
    parameter_ad,
)
phases = solve_phases(problem, algorithm)
```

All three wrappers initially require an array-valued dynamical state. Solver return
codes, statistics, and the original SciML solution are retained in
`solver_details(solution)`. An ODE solve that reaches its horizon successfully but
does not meet the state-gradient residual tolerance has `solution.converged == false`.

Extra wrapper keywords are forwarded to SciML's `solve`. `ODERelaxation` reserves a
convergence callback and combines it with a user-provided `callback` when present.
