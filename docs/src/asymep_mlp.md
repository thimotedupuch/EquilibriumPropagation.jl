# Feedforward MLP with AsymEP

`examples/asymep_mlp.jl` trains a feedforward multilayer perceptron on two interleaved
spirals. Unlike the continuous Hopfield tutorials, its connections are directed and
its dynamics do not derive from a scalar energy.

## A feedforward network as an equilibrium system

The dynamical state contains a hidden population `h` and output population `o`:

```math
\dot h = \tanh(W_xx+b_h)-h, \qquad
\dot o = W_hh+b_o-o.
```

At equilibrium these equations are exactly the usual `2 → hidden → 2` MLP forward
pass. The state Jacobian is block triangular: hidden activity drives outputs, but
outputs do not affect hidden activity during inference. This non-reciprocity is why
classical energy-based EP is inapplicable.

The tutorial implements this vector field as an explicit Lux layer and adapts it with
`lux_dynamical_model`. Lux owns the parameter tree and fixed model state;
EquilibriumPropagation owns the evolving `(h,o)` state.

## AsymEP learning phase

AsymEP first runs the ordinary feedforward dynamics to equilibrium. During each
nudged phase it freezes the free-equilibrium Jacobian asymmetry and adds the corrective
force

```math
-2A_J(s-s_0), \qquad A_J=\frac{J_F-J_F^\mathsf{T}}{2}.
```

For this triangular MLP, that correction creates the backward hidden influence needed
for credit assignment during learning without changing the forward inference
dynamics.

## Run it

The existing examples environment contains Lux and the required AD backend:

```bash
julia --project=examples examples/asymep_mlp.jl
```

Options control the small tutorial workload:

```bash
julia --project=examples examples/asymep_mlp.jl \
    --epochs=20 --observations=200 --batch-size=10 --hidden=16 --seed=7
```

The script reports the free-equilibrium residual, loss, convergence of both corrected
nudged phases, and classification accuracy. The implementation is intentionally
small enough to expose the complete vector field and AsymEP training loop.
