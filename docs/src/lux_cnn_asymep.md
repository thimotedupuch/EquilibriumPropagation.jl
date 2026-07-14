# Train an arbitrary Lux network with AsymEP

This tutorial trains every parameter of a small VGG-style convolutional network
on Fashion-MNIST.  It demonstrates a useful generic construction: any
**state-invariant** Lux predictor can be placed inside an AsymEP or Dyadic EP
problem without rewriting its layers as equilibrium dynamics.

The complete runnable example is
[`examples/lux_cnn_asymep.jl`](https://github.com/thimotedupuch/EquilibriumPropagation.jl/blob/master/examples/lux_cnn_asymep.jl).

## Lift the predictor into an equilibrium system

Let ``f_\theta(x)`` be an arbitrary Lux network and let ``s`` contain its output
logits.  Define the vector field

```math
F_\theta(s, x) = f_\theta(x) - s.
```

Its free equilibrium is ``s_* = f_\theta(x)``.  Consequently, differentiating
the EP parameter objective traverses the complete Lux model, including its
convolutions.  The example constructs this field with [`lux_dynamical_model`](@ref):

```julia
function output_equilibrium_model(layer::Lux.AbstractLuxLayer)
    lux_dynamical_model(
        layer;
        layer_input=(state, input) ->
            reshape(input, 28, 28, 1, size(input, 2)),
        dynamics_output=(logits, state, input) -> logits .- state,
        cost=(state, target, ps, st) ->
            sum(abs2, state .- target) / (2 * size(state, 2)),
        readout=(state, ps, st) -> state,
        initial_state=(ps, input, st) ->
            zeros(eltype(input), 10, size(input, 2)),
    )
end
```

Only the ten logits per example are equilibrium variables, so forward-mode AD
is appropriate for state derivatives.  Parameter differentiation must traverse
the larger CNN and therefore uses reverse-mode Enzyme:

```julia
algorithm = AsymEP(
    0.1f0,
    Relaxation(dt=1f0, maxiters=12, abstol=2f-5);
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoEnzyme(function_annotation=Enzyme.Duplicated),
)
```

Replace `AsymEP` with `DyadicEP` to run the dyadic estimator.

## A compact VGG-style model

Fashion-MNIST keeps the example light enough for a laptop while still exercising
spatial feature learning.  The network uses two convolutional blocks, pooling,
global mean pooling, and a learned classifier:

```julia
layer = Lux.Chain(
    Lux.Conv((3, 3), 1 => 8, relu; pad=Lux.SamePad()),
    Lux.Conv((3, 3), 8 => 8, relu; pad=Lux.SamePad()),
    Lux.MaxPool((2, 2)),
    Lux.Conv((3, 3), 8 => 16, relu; pad=Lux.SamePad()),
    Lux.Conv((3, 3), 16 => 16, relu; pad=Lux.SamePad()),
    Lux.GlobalMeanPool(),
    Lux.FlattenLayer(),
    Lux.Dense(16 => 10),
)
```

All convolutional and dense parameters are updated by `train_step!`; unlike the
CIFAR-10 CHN tutorial, the convolutional feature extractor is not frozen.

Run the default AsymEP experiment from the repository root:

```bash
julia --project=examples examples/lux_cnn_asymep.jl
```

For Dyadic EP, or for a quicker smoke test:

```bash
julia --project=examples examples/lux_cnn_asymep.jl --method=dyadic
julia --project=examples examples/lux_cnn_asymep.jl \
    --epochs=1 --train-size=100 --test-size=100 --batch-size=10 --channels=4,8
```

The first Enzyme call can spend noticeable time compiling.  Later batches reuse
that compiled reverse pass.  The unit relaxation step is intentional: because
the free vector field is affine in the logits, it reaches the free equilibrium
in one update; the nudged phases then converge geometrically.

## Scope of the construction

This example establishes compatibility with arbitrary stateless Lux predictors,
but it should not be confused with a convolutional equilibrium network whose
intermediate feature maps are themselves dynamical variables.  Here
``\partial F/\partial s = -I``, so the state dynamics are conservative and the
asymmetry correction vanishes.  In this limiting case the EP parameter gradient
agrees with ordinary supervised differentiation through ``f_\theta``.

Reproducing the paper's full VGG-style equilibrium experiment requires lifting
every intermediate activation into ``s`` and defining the coupled, generally
asymmetric layer dynamics.  That is possible with the same [`AsymEP`](@ref) and
[`DyadicEP`](@ref) APIs—the [feedforward AsymEP MLP](asymep_mlp.md) shows the
pattern—but convolutional feature maps make the state and its Jacobian much
larger.  The present tutorial is deliberately the smaller, reproducible step:
the complete CNN is learned, while the equilibrium state remains compact.
