# MNIST classification example

The repository includes `examples/mnist_classification.jl`, a compact MNIST
classifier trained with symmetric equilibrium propagation. Images are hard-clamped
external inputs and the ten class-score neurons form the relaxed dynamical state.

To keep ForwardDiff-based parameter gradients practical, the script average-pools
each 28×28 image to 7×7 by default. It trains on a random 1,000-image subset and
reports accuracy on 500 test images.

## Run it

Instantiate the example environment once, then launch the script:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/mnist_classification.jl
```

MLDatasets may ask for permission to download MNIST the first time it is used.

The dataset size, pooling, and training duration are configurable:

```bash
julia --project=examples examples/mnist_classification.jl \
    --epochs=10 --train-samples=5000 --test-samples=1000 \
    --batch-size=20 --pool=4 --seed=11
```

`--pool=2` retains 14×14 inputs and gives the classifier more spatial detail, at the
cost of a larger parameter gradient. `--pool=1` uses all 784 pixels.

The example deliberately uses a quadratic energy so that its role is transparent:

```math
E(s, x; \theta) = \frac{1}{B}\left(\frac{1}{2}\lVert s\rVert^2
    - s^\mathsf{T}(Wx+b)\right).
```

At the free equilibrium, the output state contains the ten class scores. During the
nudged phases, a bipolar target attracts the correct class score toward `+1` and the
other scores toward `-1`.
