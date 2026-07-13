# Continuous Hopfield example

The repository includes `examples/continuous_hopfield_spiral.jl`, a complete
two-class training example.

The example declares a `ContinuousHopfield((2, hidden, 2))` specification and calls
`setup` to obtain an ordinary `EPModel` and named parameter tree. The dynamical state
contains hidden and output neurons. Its conservative energy combines quadratic neuron
potentials with tanh firing rates, input drive, hidden–output coupling, and biases.
Because all interactions appear in one scalar energy, state dynamics are obtained
directly from its gradient. Bipolar output targets provide the supervised nudging cost.

## Run it

From the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/continuous_hopfield_spiral.jl
```

The default run generates 400 noisy points, trains a 2–32–2 network with symmetric EP,
and reports loss, accuracy, free-phase residual, and whether every training phase in
the epoch converged.

Command-line options use `--name=value` syntax:

```bash
julia --project=examples examples/continuous_hopfield_spiral.jl \
    --epochs=50 --points=400 --batch-size=20 --hidden=48 --seed=11
```

`points` must be even and divisible by `batch-size`. The script returns the trained
model, named parameter tree, data, labels, and final accuracy when `train_spiral` is
called programmatically.

## Why the step scales with batch size

Both energy and cost are means over the minibatch. Their gradient with respect to each
neuron therefore contains a factor of `1 / batch_size`. The example scales the Euler
step by `batch_size`, keeping the effective per-neuron relaxation step unchanged when
the batch size changes.
