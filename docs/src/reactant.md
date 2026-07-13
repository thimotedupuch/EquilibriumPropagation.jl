# OpenXLA acceleration with Reactant

The optional Reactant extension compiles the expensive parts of batched conservative
EP—free and nudged relaxation, EnzymeMLIR differentiation, residuals, and optionally a
plain-SGD update—into one OpenXLA executable. The same API targets CPU, GPU, and TPU.
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
    learning_rate=1f-2,
)

step = compile_reactant(problem, algorithm; backend="cpu")
result = step()
```

`result` contains `gradient`, updated `parameters`, all phase states, the free-phase
`loss`, and terminal `free_residual`, `positive_residual`, and `negative_residual`.
If `learning_rate=nothing`, `result.parameters` is unchanged. Optimizer state is not
hidden inside the executable; the initial compiled update is intentionally plain SGD.

Backend selection occurs before device transfer. Choose `"cpu"`, `"gpu"`, or
`"tpu"`; the corresponding OpenXLA runtime and hardware must be available. The first
call to `compile_reactant` pays tracing and compilation cost, so reuse its result.

## Reuse without recompiling

Reactant specializes the executable to argument shapes and traced Julia control flow.
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

## Current compilation contract

The first extension is deliberately small and predictable:

- the model must be a conservative [`EPModel`](@ref);
- the protocol must be [`OneSidedEP`](@ref) or [`SymmetricEP`](@ref);
- the dynamical state must be one numeric array;
- input, target, state, and parameter shapes are fixed after compilation;
- each phase always performs its configured number of explicit Euler steps; and
- the energy, cost, and model functions must be traceable by Reactant and
  differentiable by EnzymeMLIR.

Terminal residuals reveal whether the fixed step budget was sufficient. Dynamic
early stopping, adaptive/SciML solvers, tree-valued states, Continuous EP,
Holomorphic EP, AsymEP, and Dyadic EP are not part of this initial compiled path.

## Complete example

Run the accelerated two-class training example from the repository root:

```bash
julia --project=examples examples/reactant_ep.jl --backend=cpu
```

Its other options are `--epochs=12`, `--observations=256`, `--batch-size=32`, and
`--seed=7`. Substitute `--backend=gpu` or `--backend=tpu` on configured hardware.
