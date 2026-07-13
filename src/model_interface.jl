"""
    EPModel(; energy, cost, readout, initial_state)

Function-based conservative EP model. `energy` and `cost` must return scalars;
`initial_state` constructs the dynamical state; and `readout` maps an equilibrium
state to a prediction. The alias keyword `initialstate` is also accepted.

Functions are called as `energy(s, ps, x, st)`, `cost(s, y, ps, st)`,
`readout(s, ps, st)`, and `initial_state(ps, x, st)`.
"""
struct EPModel{E,C,R,I}
    energy::E
    cost::C
    readout::R
    initial_state::I
end

"""
    DynamicalModel(; dynamics, cost, readout, initial_state)

Function-based model for arbitrary differentiable, potentially non-conservative
dynamics. `dynamics(s, ps, x, st)` returns `ds/dt`; unlike [`EPModel`](@ref), it need
not derive from a scalar energy. The remaining call signatures match `EPModel`.
"""
struct DynamicalModel{D,C,R,I}
    dynamics::D
    cost::C
    readout::R
    initial_state::I
end

function DynamicalModel(; dynamics, cost, readout, initial_state=nothing,
                        initialstate=nothing)
    init = initial_state === nothing ? initialstate : initial_state
    init === nothing && throw(ArgumentError("initial_state is required"))
    return DynamicalModel(dynamics, cost, readout, init)
end

function EPModel(; energy, cost, readout, initial_state=nothing, initialstate=nothing)
    init = initial_state === nothing ? initialstate : initial_state
    init === nothing && throw(ArgumentError("initial_state is required"))
    return EPModel(energy, cost, readout, init)
end

"""
    EPProblem(model, parameters, model_state, input, target)

Bind a model to the parameters, immutable-during-solve model state, input, and target
needed for equilibrium propagation.
"""
struct EPProblem{M,P,S,X,Y}
    model::M
    parameters::P
    model_state::S
    input::X
    target::Y
end

"""Return a model's scalar internal energy at `state`."""
energy(model::EPModel, state, parameters, model_state, input) =
    model.energy(state, parameters, input, model_state)

"""Return a model's scalar supervised cost at `state`."""
cost(model::EPModel, state, parameters, model_state, target) =
    model.cost(state, target, parameters, model_state)

"""Map a dynamical state to the model prediction used for inference."""
readout(model::EPModel, state, parameters, model_state) =
    model.readout(state, parameters, model_state)

"""Construct the initial dynamical state for an equilibrium solve."""
initial_state(model::EPModel, parameters, model_state, input) =
    model.initial_state(parameters, input, model_state)

"""Evaluate a [`DynamicalModel`](@ref)'s vector field at `state`."""
vector_field(model::DynamicalModel, state, parameters, model_state, input) =
    model.dynamics(state, parameters, input, model_state)

cost(model::DynamicalModel, state, parameters, model_state, target) =
    model.cost(state, target, parameters, model_state)

readout(model::DynamicalModel, state, parameters, model_state) =
    model.readout(state, parameters, model_state)

initial_state(model::DynamicalModel, parameters, model_state, input) =
    model.initial_state(parameters, input, model_state)

"""
    predict(solution, model, parameters, model_state)
    predict(solution, problem)

Apply the model readout to an equilibrium solution.
"""
predict(solution::EquilibriumSolution, model, parameters, model_state) =
    readout(model, solution.state, parameters, model_state)

predict(solution::EquilibriumSolution, problem::EPProblem) =
    predict(solution, problem.model, problem.parameters, problem.model_state)
