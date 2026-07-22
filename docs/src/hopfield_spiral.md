# Continuous Hopfield example

This example trains a small continuous Hopfield network on two spirals. It is the
simplest complete classification example in the repository.

## 1. Load the packages and data

The example file contains `spiral_dataset`, which returns inputs, bipolar targets,
and integer labels.

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff
using Optimisers
using Random

include("examples/continuous_hopfield_spiral.jl")

rng = Xoshiro(7)
inputs, targets, labels = spiral_dataset(rng, 400)
```

Each column is one observation. `inputs` and `targets` both have two rows.

## 2. Build the model

```julia
network = ContinuousHopfield(
    (2, 32, 2);
    activation=tanh,
    output_cost=BipolarSquaredError(),
)

model, parameters = EquilibriumPropagation.setup(rng, network)
```

The input is clamped. The hidden and output neurons relax to an equilibrium.

## 3. Choose EP and an optimizer

```julia
backend = AutoForwardDiff()
algorithm = EPAlgorithm(
    SymmetricEP(0.15f0),
    Relaxation(dt=7f0, maxiters=50, abstol=5f-4);
    state_ad=backend,
    parameter_ad=backend,
)

optimizer_state = Optimisers.setup(Optimisers.Adam(1f-2), parameters)
```

The energy is averaged over a batch of 20. The larger `dt` offsets that averaging.

## 4. Train on one batch

```julia
columns = 1:20
batch = (inputs[:, columns], targets[:, columns])

optimizer_state, parameters, stats = train_step!(
    optimizer_state,
    parameters,
    model,
    batch,
    algorithm,
)

@assert stats.converged
```

Put this call inside an epoch and minibatch loop to train the full dataset. The
repository script already does that.

## Run the complete script

```bash
julia --project=examples examples/continuous_hopfield_spiral.jl \
    --epochs=30 --points=400 --batch-size=20
```

It prints loss, accuracy, residual, and convergence while it trains.
