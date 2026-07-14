# EP concepts

## Total energy and phases

For state ``s``, parameters ``\theta``, input ``x``, target ``y``, and nudging
strength ``\beta``, write the total energy, following the usual EP notation, as

```math
F_\beta(s; \theta, x, y)
= E(s; \theta, x) + \beta C(s, y; \theta).
```

The built-in relaxation follows the state gradient flow with an explicit Euler step:

```math
s_{k+1} = s_k - \Delta t\,\nabla_s F_\beta(s_k;\theta,x,y).
```

The free phase uses ``\beta=0``. Nudged phases start from the converged free state,
which generally reduces solve time and ensures the phase contrast follows the same
nearby equilibrium branch.

## Gradient estimators

[`OneSidedEP`](@ref) uses the contrast between a positive and free phase:

```math
g_\theta \approx
\frac{\partial_\theta F_\beta(s_\beta;\theta,x,y)
- \partial_\theta F_0(s_0;\theta,x,y)}{\beta}.
```

[`SymmetricEP`](@ref) uses positive and negative nudging:

```math
g_\theta \approx
\frac{\partial_\theta F_{+\beta}(s_{+\beta};\theta,x,y)
- \partial_\theta F_{-\beta}(s_{-\beta};\theta,x,y)}{2\beta}.
```

Symmetric EP cancels the leading finite-``\beta`` error and is the recommended
default when the negative phase remains stable.

In both estimators, equilibrated states are constants during parameter
differentiation. DifferentiationInterface only sees the parameters as active input;
no relaxation trajectory is recorded or differentiated.

## Selecting AD backends

`EPAlgorithm` stores independent `state_ad` and `parameter_ad` selectors. This is
useful because state gradients and parameter gradients can have different shapes,
mutation patterns, and best AD modes.

```julia
using ADTypes: AutoForwardDiff
using ForwardDiff

algorithm = EPAlgorithm(
    SymmetricEP(0.05),
    Relaxation();
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoForwardDiff(),
)
```

ADTypes describes the backend; DifferentiationInterface performs the operation. The
backend implementation must be installed and loaded by user code. Relaxation prepares
the state-gradient operation once per phase and reuses that preparation across steps.

## Convergence and approximation error

An EP gradient is affected independently by finite nudging, incomplete equilibrium,
and differentiation error. [`EPStats`](@ref) reports the evidence needed to distinguish
them:

- phase residuals, energies, and iteration counts;
- phase convergence flags;
- gradient norm; and
- displacement of each nudged phase from the free state.

Decrease ``\beta`` to study finite-nudging error, and tighten the relevant phase
solver to study equilibrium error. A smaller ``\beta`` generally requires more
accurate equilibria because phase differences also become smaller.

## Model-state restriction

`model_state` is passed to every model function but must remain constant during a
solve. Operations such as dropout or updates to normalization statistics would change
the energy landscape while it is being minimized and violate the solver contract.
