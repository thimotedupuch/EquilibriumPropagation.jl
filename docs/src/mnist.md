# MNIST classification example

This example changes the spiral classifier in only two places: the data and the
network size.

## 1. Load a small MNIST dataset

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff
using Optimisers
using Random

include("examples/mnist_classification.jl")

inputs, targets, labels, test_inputs, _, test_labels = load_mnist(
    train_samples=1_000,
    test_samples=500,
    pool=4,
)
```

Pooling turns each 28×28 image into 49 input values. Each target has ten bipolar
values: `+1` for the correct digit and `-1` for the others.

## 2. Build the classifier

```julia
rng = Xoshiro(7)
network = ContinuousHopfield(
    (size(inputs, 1), 64, 10);
    activation=tanh,
    output_cost=BipolarSquaredError(),
)

model, parameters = EquilibriumPropagation.setup(rng, network)
```

## 3. Configure training

```julia
batch_size = 20
backend = AutoForwardDiff()

algorithm = EPAlgorithm(
    SymmetricEP(0.1f0),
    Relaxation(dt=0.8f0 * batch_size, maxiters=20, abstol=1f-5);
    state_ad=backend,
    parameter_ad=backend,
)

optimizer_state = Optimisers.setup(Optimisers.Adam(1f-2), parameters)
```

## 4. Update one minibatch

```julia
columns = 1:batch_size
optimizer_state, parameters, stats = train_step!(
    optimizer_state,
    parameters,
    model,
    (inputs[:, columns], targets[:, columns]),
    algorithm,
)
```

Repeat this call for shuffled minibatches. Watch `stats.loss` and
`stats.converged`.

## Run the complete script

```bash
julia --project=examples examples/mnist_classification.jl
```

The defaults are deliberately small. Use `--train-samples`, `--test-samples`,
`--epochs`, `--hidden`, and `--pool` to scale the experiment.
