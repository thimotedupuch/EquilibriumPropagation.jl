# API reference

## Models and problems

```@docs
EPModel
EPProblem
energy
cost
readout
initial_state
predict
augmented_energy
```

## Protocols and algorithms

```@docs
FreePhase
NudgedPhase
OneSidedEP
SymmetricEP
Relaxation
EPAlgorithm
```

## Solving and gradients

```@docs
equilibrate
solve_phases
ep_gradient
```

## Results and diagnostics

```@docs
EquilibriumSolution
EPPhases
EPStats
```

## Optimisers extension

```@docs
apply_gradient!
train_step!
```
