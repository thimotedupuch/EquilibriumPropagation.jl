# OpenXLA acceleration with Reactant

The optional Reactant extension compiles conservative and non-conservative EP
relaxation, EnzymeMLIR differentiation, Jacobian operations, residuals, and optimizer
updates into OpenXLA executables. The same API targets CPU, GPU, and TPU.
Reactant and Enzyme remain optional dependencies of EquilibriumPropagation.jl.

Install them in the environment containing your application:

```julia
using Pkg
Pkg.add(["Reactant", "Enzyme"])
```

Then load all three packages:

```julia
using EquilibriumPropagation
using Enzyme
using Reactant
```

The implementation uses Reactant's tracing compiler and EnzymeMLIR reverse-mode AD.
See the upstream [Reactant API](https://enzymead.github.io/Reactant.jl/stable/api/api)
and [automatic differentiation tutorial](https://enzymead.github.io/Reactant.jl/stable/tutorials/automatic-differentiation)
for the underlying compiler and AD interfaces.

## Compile a fixed-shape EP step

Array batches follow the `features × batch` convention:

```julia
batch = EPBatch(input, target)
problem = EPProblem(model, parameters, model_state, batch)

algorithm = ReactantEP(
    SymmetricEP(0.15f0);
    dt=8.0f0,
    free_steps=12,
    nudged_steps=8,
    abstol=1f-5,
    reltol=1f-4,
    learning_rate=1f-2,
)

step = compile_reactant(problem, algorithm; backend="cpu")
result = step()
```

`result` contains `gradient`, updated `parameters`, all phase states, the free-phase
`loss`, and terminal `free_residual`, `positive_residual`, and `negative_residual`.
It also reports per-phase `*_iterations` and `*_converged` values plus an aggregate
`converged` flag. With the default `abstol=0` and `reltol=0`, every phase performs
its complete step budget. Nonzero tolerances use
`abstol + reltol * initial_residual`; after a phase meets that threshold, its state is
terminated by a bounded Reactant `@trace while`, so the device stops executing that
phase once it converges while retaining a hard maximum step count.
If `learning_rate=nothing`, `result.parameters` is unchanged. Optimizer state is not
hidden inside this basic executable; its built-in update is intentionally plain SGD.

Backend selection occurs before device transfer. Choose `"cpu"`, `"gpu"`, or
`"tpu"`; the corresponding OpenXLA runtime and hardware must be available. The first
call to `compile_reactant` pays tracing and compilation cost, so reuse its result.

## Reuse without recompiling

Reactant specializes the executable to argument shapes, tree structure, and traced
Julia control flow.
Transfer equal-shaped problems once, retain parameters on the device, and reuse the
four-argument executable call:

```julia
prepared = map(problem -> reactant_inputs(problem; backend="cpu"), problems)
device_parameters = first(prepared).parameters

for inputs in prepared
    result = step(
        device_parameters,
        inputs.model_state,
        inputs.batch,
        inputs.initial_state,
    )
    device_parameters = result.parameters
end
```

This avoids recompiling each minibatch and avoids copying updated parameters back to
the host. Host conversion, for logging or checkpoints, is explicit:

```julia
host_parameters = Array(device_parameters)
```

Keep every minibatch the same size. Either discard/pad a final remainder or compile a
second executable for its shape. Energy and cost functions must return scalars and
should reduce over the batch explicitly; batch means make learning-rate behavior
independent of batch size. If the energy is a batch mean, scale `dt` with batch size
to retain the desired per-state Euler step.

## Existing EPAlgorithm values

An ordinary `EPAlgorithm` can be compiled directly when all three phase solvers are
`Relaxation` instances with compatible step sizes and tolerances:

```julia
step = compile_reactant(problem, algorithm; backend="cpu")
```

The AD selectors stored in `EPAlgorithm` are replaced by EnzymeMLIR inside the
compiled program. Adaptive SciML solvers are not interchangeable with this path; see
the limitations below.

## Optimisers.jl and Continuous EP

Loading Optimisers.jl activates a three-argument form. Pass an optimizer rule rather
than a host `Optimisers.setup` result:

```julia
using Optimisers

step = compile_reactant(
    problem,
    ReactantEP(SymmetricEP(0.1f0); dt=0.2f0),
    Optimisers.Adam(1f-3);
    backend="cpu",
)
result = step()
```

Optimizer setup and updates execute on the device. `result.optimizer_state` can be
fed back alongside `result.parameters` through the five-argument executable call,
preserving momentum, Adam moments, counters, and optimizer chains without a host
round trip. Set `ReactantEP.learning_rate=nothing` when supplying an optimizer.

The same form compiles continual-update EP:

```julia
step = compile_reactant(
    problem,
    continuous_algorithm,
    Optimisers.Descent(1f-3);
    backend="cpu",
)
```

The free phase, every local EP gradient, learning/nudging schedules, optimizer state,
and coupled state/parameter updates are enclosed in the executable. Schedule
functions must be Reactant-traceable and free of host side effects.

## Non-conservative models

`DynamicalModel` supports compiled `AsymEP` and `DyadicEP`:

```julia
asym_step = compile_reactant(problem, asym_algorithm; backend="cpu")
dyadic_step = compile_reactant(problem, dyadic_algorithm; backend="cpu")
```

The extension constructs field Jacobians from Enzyme forward-mode directional
derivatives because Enzyme's high-level tuple-stacking `jacobian` helper is not
lowerable by Reactant 0.2. AsymEP returns its positive and negative states; DyadicEP
returns midpoint and difference states. Both return the gradient, loss, residuals,
iteration counts, and Jacobian-asymmetry diagnostic. Their current package-level
array-state restriction still applies.

## Current compilation contract

The extension keeps a deliberately predictable compilation contract:

- conservative models support [`OneSidedEP`](@ref) and [`SymmetricEP`](@ref);
- non-conservative `DynamicalModel` values support [`AsymEP`](@ref) and
  [`DyadicEP`](@ref);
- the dynamical state may be a numeric array or a Functors-compatible tree whose
  leaves are numeric arrays or tracked numbers;
- input, target, state, parameter shapes, and tree structures are fixed after
  compilation;
- each phase has a fixed maximum number of Euler, RK4, or damped-Newton steps, with
  optional tolerance-based masking; and
- the energy, cost, and model functions must be traceable by Reactant and
  differentiable by EnzymeMLIR.

Terminal residuals and convergence flags reveal whether the maximum step budget was
sufficient.

For a higher-order compiled solve, select classical fourth-order Runge--Kutta:

```julia
algorithm = ReactantEP(
    SymmetricEP(0.1f0); method=:rk4, dt=0.2f0,
    free_steps=100, nudged_steps=50,
)
```

RK4 evaluates the state gradient four times per step but has a substantially larger
stability region and better trajectory accuracy than Euler. Both methods, including
their bounded convergence loop, are contained in the reusable OpenXLA executable.

For a dedicated steady-state solve, use dense damped Newton:

```julia
algorithm = ReactantEP(
    SymmetricEP(0.1f0);
    method=:newton,
    free_steps=20,
    nudged_steps=10,
    abstol=1f-6,
    reltol=1f-6,
    damping=1f-4,
    step_scale=1f0,
)
```

Newton solves ``(H + \lambda I)\,\Delta = \nabla_s E`` and updates
``s \leftarrow s - \alpha\Delta``. Enzyme constructs the dense state Hessian and
Reactant lowers the factorization and bounded convergence loop to OpenXLA. Damping
regularizes singular or poorly conditioned Hessians; `step_scale` can be reduced below
one when full Newton steps are too aggressive.

Dense Newton currently requires an array-valued state and uses quadratic memory in
the number of state elements. It is intended for modest states; Euler and RK4 remain
the appropriate compiled methods for large states until a matrix-free Krylov method
is available.

Two package workflows cannot currently be enclosed safely:

- `HolomorphicEP`: Reactant 0.2 leaves complex `enzyme.batch` operations unlowered
  when optimization is disabled, while its optimized complex `dot_general` rewrite
  can crash natively. `compile_reactant` rejects this combination before compilation.
- adaptive SciML solvers (`ODERelaxation`, `SteadyStateRelaxation`, and
  `RootRelaxation`): their integrator orchestration, callbacks, dynamic caches, and
  solver objects execute on the Julia host rather than as StableHLO. Their model RHS
  operations can be accelerated independently, but the whole solver cannot be part
  of this reusable executable. Use `Relaxation` for whole-step compilation.

Changing input shapes, batch sizes, parameter shapes, or tree structure still
requires recompilation, as these are part of XLA specialization. Mutable Lux model
state remains unsupported because it violates the package's equilibrium-model
contract independently of Reactant.

## Tree-valued states and parameters

Reactant recursively transfers ordinary tuples, named tuples, and types supported by
Functors.jl. The energy, cost, and initial-state functions can therefore use structured
states and parameter trees directly:

```julia
initial_state = (ps, x, st) -> (
    hidden=zero(ps.input * x),
    output=zeros(eltype(x), 2, size(x, 2)),
)

energy = (s, ps, x, st) ->
    sum(abs2, s.hidden) / 2 + sum(abs2, s.output) / 2 -
    sum(s.hidden .* (ps.input * x)) -
    sum(s.output .* (ps.readout * s.hidden))
```

The compiled `gradient`, phase states, and updated `parameters` retain the same tree
structure. All leaves must remain traceable by Reactant and differentiable by
EnzymeMLIR.

## Complete example

Run the accelerated two-class training example from the repository root:

```bash
julia --project=examples examples/reactant_ep.jl --backend=cpu
```

Its other options are `--epochs=12`, `--observations=256`, `--batch-size=32`, and
`--seed=7`. Substitute `--backend=gpu` or `--backend=tpu` on configured hardware.

For a larger image-classification pipeline, continue with
[CIFAR-10 with Reactant](@ref). It composes a
Reactant-compiled Lux feature extractor with compiled CHN relaxation, symmetric EP,
and Adam while keeping intermediate feature tensors on the accelerator.
