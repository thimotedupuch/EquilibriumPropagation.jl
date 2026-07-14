# CIFAR-10 with frozen Lux convolutions and a compiled CHN

This larger example classifies CIFAR-10 images with a two-stage OpenXLA pipeline:

1. a randomly initialized, frozen Lux convolutional network maps each
   `32 × 32 × 3` image to a compact feature vector; and
2. a continuous Hopfield network (CHN) is trained on those features with symmetric
   equilibrium propagation and Adam.

Both stages are compiled with Reactant. The convolutions, pooling, CHN relaxation,
EnzymeMLIR state and parameter derivatives, EP phase contrast, and Adam update execute
on the selected accelerator. The convolution parameters are intentionally never
trained: this isolates EP training to the energy-based CHN and avoids silently turning
the example into end-to-end backpropagation.

The complete program is
[`examples/reactant_cifar10_chn.jl`](https://github.com/thimotedupuch/EquilibriumPropagation.jl/blob/master/examples/reactant_cifar10_chn.jl).

## Run the example

Instantiate the examples environment and start with the smaller default configuration:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/reactant_cifar10_chn.jl --backend=cpu
```

MLDatasets downloads CIFAR-10 on the first run. The defaults use 4,096 training
images, 1,024 test images, 32 convolutional features, and a `32–128–10` CHN. They are
intended as a tutorial-sized run, not as a competitive CIFAR benchmark.

On a configured GPU, a larger run is:

```bash
julia --project=examples examples/reactant_cifar10_chn.jl \
  --backend=gpu --epochs=30 --train-size=50000 --test-size=10000 \
  --batch-size=100 --channels=64 --hidden=512
```

All sizes must be divisible by the batch size because XLA specializes executables to
static tensor shapes. Compilation is substantial, especially on CPU; it is paid once
for the convolutional extractor and once for the complete CHN training step.

## Frozen random convolutional features

The extractor uses two strided Lux convolutions, mean pooling, global mean pooling,
and flattening:

```julia
Lux.Chain(
    Lux.WrappedFunction(x -> 2f0 .* x .- 1f0),
    Lux.Conv((5, 5), 3 => 16, tanh; stride=2, pad=Lux.SamePad()),
    Lux.MeanPool((2, 2)),
    Lux.Conv((3, 3), 16 => 32, tanh; stride=2, pad=Lux.SamePad()),
    Lux.GlobalMeanPool(),
    Lux.FlattenLayer(),
)
```

The first operation maps CIFAR pixels from `[0, 1]` to `[-1, 1]`. After Lux setup,
the parameters and test-mode state are transferred to Reactant and the extractor is
compiled. Each fixed-size image batch is then converted into a device-resident
`features × batch` tensor.

`FlattenLayer` normally exposes a lazy reshaped view. The example materializes that
view with a fused zero-add inside the compiled extractor, yielding an ordinary device
tensor that can be passed directly into `reactant_inputs` without a host round trip.

Features are cached because the convolutional filters remain fixed. This makes later
epochs spend their time entirely in CHN equilibrium relaxation and learning.

## The continuous Hopfield classifier

For the default feature width, the energy model has layer sizes `(32, 128, 10)`:

```julia
network = ContinuousHopfield(
    (32, 128, 10);
    activation=tanh,
    potential=QuadraticPotential(),
    input_clamp=HardClamp(),
    output_cost=BipolarSquaredError(),
    recurrent=false,
)
model, parameters = EquilibriumPropagation.setup(rng, network)
```

The ten output neurons receive bipolar targets: the correct class is `+1` and every
other class is `-1`. Hidden and output neurons are both dynamical state variables;
the convolutional feature vector is hard-clamped input.

Because the CHN energy and cost are batch means, the Euler step scales with batch
size. Nonzero tolerances activate bounded `@trace while` convergence loops:

```julia
algorithm = ReactantEP(
    SymmetricEP(0.12f0);
    dt=0.28f0 * batch_size,
    free_steps=30,
    nudged_steps=15,
    abstol=4f-3,
    reltol=1f-4,
)

step = compile_reactant(
    first(train_problems),
    algorithm,
    Optimisers.Adam(2f-3);
    backend,
)
```

The returned executable owns the initial device parameters and Adam state. Training
feeds both values returned by one minibatch directly into the next call, avoiding
parameter or optimizer-state transfers back to Julia.

## What this example demonstrates

- Lux layers can serve as compiled, frozen feature transforms without becoming part
  of the EP dynamical state.
- Device outputs from one Reactant executable can become inputs to another.
- A structured CHN parameter tree and Adam's moment/counter state remain on device.
- Free and nudged convergence loops terminate dynamically while retaining fixed
  maximum bounds.
- Test accuracy uses the compiled free-phase state; only the small output score matrix
  is copied to the host for `argmax` and reporting.

Random convolutional features deliberately trade accuracy for a clean separation of
mechanisms. Increasing `channels`, `hidden`, data size, and epochs generally provides
a stronger experiment, but it also increases equilibrium-state memory and compilation
cost. This tutorial is meant to demonstrate accelerator composition rather than claim
state-of-the-art CIFAR-10 performance.
