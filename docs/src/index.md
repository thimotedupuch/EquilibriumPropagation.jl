# EquilibriumPropagation.jl

EquilibriumPropagation.jl implements the core equilibrium propagation workflow for
conservative energy-based systems:

1. define a scalar energy, supervised cost, readout, and initial state;
2. find a free equilibrium;
3. warm-start one or two nudged equilibria;
4. contrast phase-local parameter derivatives; and
5. update parameters with an Optimisers.jl rule or custom training code.

The package is deliberately not a neural-network framework or a general implicit
layer library. Models are ordinary Julia functions, AD implementations are selected
with ADTypes, and equilibrium solving remains separate from the EP protocol.

## Package boundary

Ordinary EP training differentiates the augmented energy with respect to parameters
while treating each equilibrated state as constant. It does **not** differentiate
through relaxation. This keeps the implementation faithful to the EP learning rule
and prevents it from silently becoming an implicit-gradient method.

## Current capabilities

- [`OneSidedEP`](@ref) and [`SymmetricEP`](@ref)
- warm-started free, positive, and negative phases
- [`Relaxation`](@ref) with independent phase configurations
- state and parameter differentiation through DifferentiationInterface
- structured phase and gradient diagnostics in [`EPStats`](@ref)
- optional Optimisers.jl integration through [`train_step!`](@ref)

Start with [Getting started](@ref), then read [EP concepts](@ref) for the
mathematical and numerical contracts.
