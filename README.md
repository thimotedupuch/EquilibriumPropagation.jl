# EquilibriumPropagation.jl

EquilibriumPropagation.jl is a composable Julia toolkit for training equilibrium
models with equilibrium propagation. It supports conservative energy-based networks,
non-conservative dynamical systems, continuous and holomorphic EP variants, multiple
equilibrium solvers, Lux models, Optimisers.jl, and Reactant/OpenXLA acceleration.

The package keeps models, learning protocols, solvers, differentiation backends, and
optimizer state explicit. Parameter gradients are evaluated at equilibria without
differentiating through the relaxation trajectory.

## Documentation

The complete documentation is available at:

**[thimotedupuch.github.io/EquilibriumPropagation.jl/dev](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/)**

Useful entry points:

- [Getting started](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/getting_started/)
- [EP concepts](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/concepts/)
- [Equilibrium solvers](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/solvers/)
- [Continuous EP](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/continuous_ep/)
- [Non-conservative EP](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/nonconservative_ep/)
- [Holomorphic EP](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/holomorphic_ep/)
- [Lux integration](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/lux/)
- [Reactant/OpenXLA acceleration](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/reactant/)
- [API reference](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/api/)

## Tutorials

- [Continuous Hopfield network on interleaved spirals](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/hopfield_spiral/)
- [MNIST classification with a continuous Hopfield network](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/mnist/)
- [Feedforward Lux MLP trained with AsymEP](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/asymep_mlp/)
- [CIFAR-10 with frozen Lux convolutions and a Reactant-compiled CHN](https://thimotedupuch.github.io/EquilibriumPropagation.jl/dev/reactant_cifar10/)

Runnable versions live in the [`examples/`](examples) directory.

## Installation

From Julia's package prompt:

```julia
pkg> add https://github.com/thimotedupuch/EquilibriumPropagation.jl
```

To work from a checkout:

```julia
pkg> dev /path/to/EquilibriumPropagation.jl
```

Load an implementation for the AD backend selected by your algorithm. For example:

```julia
using ADTypes: AutoForwardDiff
using EquilibriumPropagation
using ForwardDiff
```

Optional integrations are activated by loading their packages, such as Lux,
Optimisers, Reactant and Enzyme, or the relevant SciML solver packages. Installation
details and complete examples are kept in the documentation linked above.

## Development

Instantiate and run the explicit test environment with:

```bash
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
```

The Reactant and larger data examples use the separate `examples` environment.
