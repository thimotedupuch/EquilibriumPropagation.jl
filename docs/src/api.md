# API reference

## Models and problems

```@docs
EPModel
DynamicalModel
EPProblem
EPBatch
batch_size
energy
vector_field
cost
readout
initial_state
predict
augmented_energy
lux_setup
lux_energy_model
lux_dynamical_model
```

## Continuous Hopfield construction

```@docs
ContinuousHopfield
setup
QuadraticPotential
HardClamp
BipolarSquaredError
ZeroState
GlorotUniform
pack_parameters
unpack_parameters
```

## Protocols and algorithms

```@docs
FreePhase
NudgedPhase
OneSidedEP
SymmetricEP
HolomorphicEP
Relaxation
ODERelaxation
SteadyStateRelaxation
RootRelaxation
EPAlgorithm
ContinuousEP
AsymEP
DyadicEP
ReactantEP
```

## Solving and gradients

```@docs
equilibrate
solve_phases
ep_gradient
continuous_ep
continuous_train_step!
reactant_inputs
compile_reactant
```

## Results and diagnostics

```@docs
EquilibriumSolution
solver_details
EPPhases
EPStats
ContinuousEPStats
NonConservativeStats
HolomorphicPhases
HolomorphicEPStats
```

## Optimisers extension

```@docs
apply_gradient!
train_step!
```
