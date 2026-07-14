# Continuous EP

Continuous EP performs small parameter updates throughout the nudged phase instead of
waiting for a second equilibrium and applying one global update. The implementation
uses the continual-update rule of [Ernoult et al.](https://arxiv.org/abs/2005.04168):
the ordinary EP parameter-gradient contrast is written as a telescoping sum over
consecutive nudged states.

For state step ``t``, current parameters ``\theta_t``, and total energy
``F_\beta(s;\theta)=E(s;\theta)+\beta C(s;\theta)``, the package alternates

```math
s_{t+1} = s_t - \epsilon_s\,\partial_s F_{\beta_t}(s_t;\theta_t)
```

and a parameter update based on

```math
g_t = \frac{
    \partial_\theta F_{\beta_t}(s_{t+1};\theta_t)
    - \partial_\theta F_{\beta_{t-1}}(s_t;\theta_t)
}{\beta_t}.
```

The first step uses ``\beta_0=0``, so parameter-dependent supervised costs are
included correctly. With constant nudging and sufficiently slow parameter updates,
the sum of the local gradients approaches the standard one-sided EP gradient.

## Optimisers integration

```julia
using Optimisers

algorithm = ContinuousEP(
    0.05,
    Relaxation(dt=0.2, maxiters=100, abstol=1e-6, reltol=1e-4);
    state_ad=AutoForwardDiff(),
    parameter_ad=AutoForwardDiff(),
    parameter_abstol=1e-8,
    parameter_reltol=1e-4,
)

optimizer_state = Optimisers.setup(Optimisers.Adam(1e-4), parameters)
optimizer_state, parameters, stats = continuous_train_step!(
    optimizer_state,
    EPProblem(model, parameters, model_state, input, target),
    algorithm,
)
```

The optimizer is applied once per nudged state step. This is intentionally different
from `train_step!`, which applies one update after both standard EP phases.

## Schedules

`nudging_schedule(iteration, β)` chooses a finite, nonzero nudging value with the same
sign as the base ``\beta``. `learning_rate_schedule(iteration)` returns a nonnegative
factor multiplying the local gradient before it reaches the optimizer:

```julia
algorithm = ContinuousEP(
    0.05,
    solver;
    state_ad,
    parameter_ad,
    nudging_schedule=(step, β) -> β / sqrt(step),
    learning_rate_schedule=step -> 1 / step,
)
```

The constant schedules are the defaults and correspond directly to the telescoping
C-EP rule. Varying nudging includes the change in augmented energy between successive
schedule values; it no longer has the same exact finite-step telescoping identity.

## Stopping and diagnostics

The coupled phase stops only when both conditions hold:

- the state-gradient residual meets the `Relaxation` absolute/relative tolerance; and
- the latest parameter motion meets `parameter_abstol` and `parameter_reltol`.

`ContinuousEPStats` reports the free and final nudged solutions, loss, final residual,
parameter motion, total parameter displacement, separate convergence flags, and the
complete residual and parameter-motion histories. A non-converged free phase makes the
aggregate `converged` flag false even if the coupled phase itself stabilizes.

## Optimizer-neutral updates

The core API accepts any update rule without depending on Optimisers.jl:

```julia
parameters, stats = continuous_ep(problem, algorithm) do parameters, gradient, step
    fmap((parameter, derivative) -> parameter .- 1e-4 .* derivative,
         parameters, gradient)
end
```

The callback must return the parameters used by the next state step.
