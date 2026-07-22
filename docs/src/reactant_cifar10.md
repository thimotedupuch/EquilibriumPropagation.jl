# CIFAR-10 with Reactant

This example has two parts. A frozen Lux network extracts image features. A
continuous Hopfield network learns from those features. Reactant compiles both parts.

## 1. Build the feature extractor

```julia
extractor = Lux.Chain(
    Lux.WrappedFunction(x -> 2f0 .* x .- 1f0),
    Lux.Conv((5, 5), 3 => 16, tanh; stride=2, pad=Lux.SamePad()),
    Lux.MeanPool((2, 2)),
    Lux.Conv((3, 3), 16 => 32, tanh; stride=2, pad=Lux.SamePad()),
    Lux.GlobalMeanPool(),
    Lux.FlattenLayer(),
)
```

Its random parameters stay fixed. The script compiles the extractor once and caches
the resulting features.

## 2. Build the EP classifier

```julia
network = ContinuousHopfield(
    (32, 128, 10);
    activation=tanh,
    output_cost=BipolarSquaredError(),
)

model, parameters = EquilibriumPropagation.setup(rng, network)
```

The 32 features are clamped inputs. The hidden and output neurons form the
equilibrium state.

## 3. Compile one training step

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
    backend="cpu",
)
```

The first call compiles the program. Reuse `step` for every batch of the same shape.
Pass its returned parameters and optimizer state into the next call.

## Run the complete script

```bash
julia --project=examples examples/reactant_cifar10_chn.jl --backend=cpu
```

The first run may download CIFAR-10. Compilation also takes time. Start with the
defaults before increasing the dataset or network size.

Use a configured accelerator with `--backend=gpu` or `--backend=tpu`. Every batch
must have the same shape.
