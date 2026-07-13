module EquilibriumPropagationLuxExt

using EquilibriumPropagation
import Lux

_state_unchanged(old_state, new_state) = isequal(old_state, new_state)

function _apply_frozen(layer, input, parameters, model_state)
    output, new_state = Lux.apply(layer, input, parameters, model_state)
    _state_unchanged(model_state, new_state) || throw(ArgumentError(
        "Lux model state changed during equilibrium evaluation; pass the test-mode " *
        "state returned by lux_setup or use a state-invariant layer",
    ))
    return output
end

struct LuxEnergyCallable{L,I,O}
    layer::L
    layer_input::I
    energy_output::O
end

function (callable::LuxEnergyCallable)(state, parameters, input, model_state)
    layer_input = callable.layer_input(state, input)
    output = _apply_frozen(callable.layer, layer_input, parameters, model_state)
    value = callable.energy_output(output, state, input)
    value isa Number || throw(ArgumentError(
        "energy_output must return a scalar; received $(typeof(value))",
    ))
    return value
end

struct LuxDynamicsCallable{L,I,O}
    layer::L
    layer_input::I
    dynamics_output::O
end

function (callable::LuxDynamicsCallable)(state, parameters, input, model_state)
    layer_input = callable.layer_input(state, input)
    output = _apply_frozen(callable.layer, layer_input, parameters, model_state)
    field = callable.dynamics_output(output, state, input)
    size(field) == size(state) || throw(DimensionMismatch(
        "Lux vector-field output has size $(size(field)); expected $(size(state))",
    ))
    return field
end

function EquilibriumPropagation.lux_setup(rng, layer::Lux.AbstractLuxLayer)
    parameters, model_state = Lux.setup(rng, layer)
    return parameters, Lux.testmode(model_state)
end

function EquilibriumPropagation.lux_energy_model(
    layer::Lux.AbstractLuxLayer;
    cost,
    readout,
    initial_state,
    layer_input=(state, input) -> state,
    energy_output=(output, state, input) -> output,
)
    callable = LuxEnergyCallable(layer, layer_input, energy_output)
    return EPModel(; energy=callable, cost, readout, initial_state)
end

function EquilibriumPropagation.lux_dynamical_model(
    layer::Lux.AbstractLuxLayer;
    cost,
    readout,
    initial_state,
    layer_input=(state, input) -> state,
    dynamics_output=(output, state, input) -> output,
)
    callable = LuxDynamicsCallable(layer, layer_input, dynamics_output)
    return DynamicalModel(; dynamics=callable, cost, readout, initial_state)
end

end
