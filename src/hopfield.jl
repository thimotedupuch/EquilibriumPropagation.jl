"""Quadratic single-neuron potential, `Φ(s) = s² / 2`."""
struct QuadraticPotential end

(::QuadraticPotential)(x) = x^2 / 2

"""Marker selecting an input that is fixed outside the dynamical state."""
struct HardClamp end

"""Mean squared error, divided by two, for bipolar output targets."""
struct BipolarSquaredError end

function (::BipolarSquaredError)(output, target)
    batch_size = ndims(output) == 1 ? 1 : size(output, 2)
    return sum((output .- target) .^ 2) / (2 * batch_size)
end

"""Default zero-valued dynamical-state initializer."""
struct ZeroState end

"""
    GlorotUniform([T=Float32])

Glorot uniform initializer. Initializers supplied to [`ContinuousHopfield`](@ref)
use the same `(rng, rows, columns)` calling convention.
"""
struct GlorotUniform{T}
    datatype::Type{T}
end

GlorotUniform() = GlorotUniform(Float32)

function (initializer::GlorotUniform{T})(rng, rows::Integer, columns::Integer) where {T}
    limit = sqrt(T(6) / T(rows + columns))
    return (T(2) .* rand(rng, T, rows, columns) .- one(T)) .* limit
end

struct ZeroBias{T}
    datatype::Type{T}
end

ZeroBias() = ZeroBias(Float32)
(initializer::ZeroBias{T})(rng, width::Integer) where {T} = zeros(T, width)

"""
    ContinuousHopfield(layer_sizes; kwargs...)

Declarative specification for a dense continuous Hopfield network. `layer_sizes`
contains the hard-clamped input width followed by every dynamical-layer width.
Call [`setup`](@ref) to construct an ordinary [`EPModel`](@ref) and its named
parameter tree.

The initial implementation supports a hard-clamped input, dense adjacent-layer
couplings, optional biases and symmetric zero-diagonal recurrent couplings. A custom
`state_initializer` is called as `(parameters, input, model_state)`; weight and
recurrent initializers are called as `(rng, rows, columns)`, and the bias initializer
as `(rng, width)`.
"""
struct ContinuousHopfield{S,A,P,I,O,B,SI,WI,BI,RI}
    layer_sizes::S
    activation::A
    potential::P
    input_clamp::I
    output_cost::O
    bias::B
    recurrent::Bool
    state_initializer::SI
    weight_initializer::WI
    bias_initializer::BI
    recurrent_initializer::RI
end

function ContinuousHopfield(
    layer_sizes::Tuple;
    activation=tanh,
    potential=QuadraticPotential(),
    input_clamp=HardClamp(),
    output_cost=BipolarSquaredError(),
    bias=true,
    recurrent=false,
    state_initializer=ZeroState(),
    weight_initializer=GlorotUniform(),
    bias_initializer=ZeroBias(),
    recurrent_initializer=weight_initializer,
)
    length(layer_sizes) >= 2 || throw(ArgumentError(
        "layer_sizes must contain an input and at least one dynamical layer",
    ))
    all(size -> size isa Integer && size > 0, layer_sizes) ||
        throw(ArgumentError("all layer sizes must be positive integers"))
    input_clamp isa HardClamp || throw(ArgumentError(
        "only HardClamp() is currently supported",
    ))
    bias isa Bool || throw(ArgumentError("bias must be true or false"))
    recurrent isa Bool || throw(ArgumentError("recurrent must be true or false"))

    sizes = map(Int, layer_sizes)
    return ContinuousHopfield(
        sizes,
        activation,
        potential,
        input_clamp,
        output_cost,
        bias,
        recurrent,
        state_initializer,
        weight_initializer,
        bias_initializer,
        recurrent_initializer,
    )
end

_dynamic_sizes(spec::ContinuousHopfield) = Base.tail(spec.layer_sizes)

function _check_feature_shape(value, width, name)
    ndims(value) in (1, 2) || throw(DimensionMismatch(
        "$name must be a vector or a features × batch matrix",
    ))
    size(value, 1) == width || throw(DimensionMismatch(
        "$name has $(size(value, 1)) features; expected $width",
    ))
    return nothing
end

_batch_size(value) = ndims(value) == 1 ? 1 : size(value, 2)

function _check_state_shape(spec, state, input)
    expected = sum(_dynamic_sizes(spec))
    _check_feature_shape(state, expected, "state")
    ndims(state) == ndims(input) || throw(DimensionMismatch(
        "state and input must both be vectors or both be matrices",
    ))
    _batch_size(state) == _batch_size(input) || throw(DimensionMismatch(
        "state and input batch sizes must match",
    ))
    return nothing
end

function _layer_views(spec, state)
    sizes = _dynamic_sizes(spec)
    starts = cumsum((1, sizes[1:end-1]...))
    if ndims(state) == 1
        return ntuple(i -> view(state, starts[i]:(starts[i] + sizes[i] - 1)), length(sizes))
    end
    return ntuple(
        i -> view(state, starts[i]:(starts[i] + sizes[i] - 1), :),
        length(sizes),
    )
end

_bias_term(bias, rates::AbstractVector) = sum(bias .* rates)
_bias_term(bias, rates) = sum(reshape(bias, :, 1) .* rates)

function _hopfield_energy(spec, state, parameters, input)
    _check_feature_shape(input, spec.layer_sizes[1], "input")
    _check_state_shape(spec, state, input)
    layers = _layer_views(spec, state)
    rates = map(layer -> spec.activation.(layer), layers)
    previous_rates = spec.activation.(input)

    value = sum(layer -> sum(spec.potential, layer), layers)
    for i in eachindex(layers)
        value -= sum(rates[i] .* (parameters.weights[i] * previous_rates))
        spec.bias && (value -= _bias_term(parameters.biases[i], rates[i]))
        if spec.recurrent
            raw = parameters.recurrent[i]
            coupling = (raw .+ transpose(raw)) ./ 2
            coupling = coupling - Diagonal(diag(coupling))
            value -= sum(rates[i] .* (coupling * rates[i])) / 2
        end
        previous_rates = rates[i]
    end
    return value / _batch_size(input)
end

function _hopfield_cost(spec, state, target)
    layers = _layer_views(spec, state)
    output = layers[end]
    _check_feature_shape(target, spec.layer_sizes[end], "target")
    ndims(output) == ndims(target) || throw(DimensionMismatch(
        "output and target must both be vectors or both be matrices",
    ))
    _batch_size(output) == _batch_size(target) || throw(DimensionMismatch(
        "output and target batch sizes must match",
    ))
    result = spec.output_cost(output, target)
    result isa Number || throw(ArgumentError("output_cost must return a scalar"))
    return result
end

function _zero_state(spec, parameters, input)
    _check_feature_shape(input, spec.layer_sizes[1], "input")
    dimensions = ndims(input) == 1 ?
        (sum(_dynamic_sizes(spec)),) :
        (sum(_dynamic_sizes(spec)), size(input, 2))
    return zeros(eltype(input), dimensions)
end

function _initial_hopfield_state(spec, parameters, input, model_state)
    state = spec.state_initializer isa ZeroState ?
        _zero_state(spec, parameters, input) :
        spec.state_initializer(parameters, input, model_state)
    _check_state_shape(spec, state, input)
    return state
end

struct HopfieldEnergy{S}
    specification::S
end

(callable::HopfieldEnergy)(state, parameters, input, model_state) =
    _hopfield_energy(callable.specification, state, parameters, input)

struct HopfieldCost{S}
    specification::S
end

(callable::HopfieldCost)(state, target, parameters, model_state) =
    _hopfield_cost(callable.specification, state, target)

struct HopfieldReadout{S}
    specification::S
end

function (callable::HopfieldReadout)(state, parameters, model_state)
    _check_feature_shape(state, sum(_dynamic_sizes(callable.specification)), "state")
    return _layer_views(callable.specification, state)[end]
end

struct HopfieldInitialState{S}
    specification::S
end

(callable::HopfieldInitialState)(parameters, input, model_state) =
    _initial_hopfield_state(callable.specification, parameters, input, model_state)

function _hopfield_model(spec)
    return EPModel(
        energy=HopfieldEnergy(spec),
        cost=HopfieldCost(spec),
        readout=HopfieldReadout(spec),
        initial_state=HopfieldInitialState(spec),
    )
end

function EPProblem(
    model::EPModel{<:HopfieldEnergy},
    parameters,
    model_state,
    input,
    target,
)
    spec = model.energy.specification
    _check_feature_shape(input, spec.layer_sizes[1], "input")
    _check_feature_shape(target, spec.layer_sizes[end], "target")
    ndims(input) == ndims(target) || throw(DimensionMismatch(
        "input and target must both be vectors or both be matrices",
    ))
    _batch_size(input) == _batch_size(target) || throw(DimensionMismatch(
        "input and target batch sizes must match",
    ))
    return EPProblem{typeof(model),typeof(parameters),typeof(model_state),typeof(input),typeof(target)}(
        model,
        parameters,
        model_state,
        input,
        target,
    )
end

"""
    setup(rng, specification::ContinuousHopfield) -> model, parameters

Initialize a continuous Hopfield specification. The returned model is an ordinary
[`EPModel`](@ref); parameters have `weights`, `biases`, and `recurrent` tuple fields.
"""
function setup(rng, spec::ContinuousHopfield)
    sizes = spec.layer_sizes
    weights = ntuple(
        i -> spec.weight_initializer(rng, sizes[i + 1], sizes[i]),
        length(sizes) - 1,
    )
    biases = spec.bias ?
        ntuple(i -> spec.bias_initializer(rng, sizes[i + 1]), length(sizes) - 1) : ()
    recurrent = if spec.recurrent
        ntuple(length(sizes) - 1) do i
            raw = spec.recurrent_initializer(rng, sizes[i + 1], sizes[i + 1])
            symmetric = (raw .+ raw') ./ 2
            symmetric - Diagonal(diag(symmetric))
        end
    else
        ()
    end
    parameters = (weights=weights, biases=biases, recurrent=recurrent)
    for i in 1:(length(sizes) - 1)
        size(parameters.weights[i]) == (sizes[i + 1], sizes[i]) ||
            throw(DimensionMismatch(
                "weight initializer returned size $(size(parameters.weights[i])) for layer $i; " *
                "expected $((sizes[i + 1], sizes[i]))",
            ))
        if spec.bias
            size(parameters.biases[i]) == (sizes[i + 1],) || throw(DimensionMismatch(
                "bias initializer returned size $(size(parameters.biases[i])) for layer $i; " *
                "expected $((sizes[i + 1],))",
            ))
        end
        if spec.recurrent
            size(parameters.recurrent[i]) == (sizes[i + 1], sizes[i + 1]) ||
                throw(DimensionMismatch(
                    "recurrent initializer returned an incompatible size for layer $i",
                ))
        end
    end
    return _hopfield_model(spec), parameters
end

"""Flatten a continuous Hopfield named parameter tree into one vector."""
function pack_parameters(spec::ContinuousHopfield, parameters)
    leaves = (parameters.weights..., parameters.biases..., parameters.recurrent...)
    isempty(leaves) && return Float64[]
    return reduce(vcat, map(vec, leaves))
end

"""Rebuild a continuous Hopfield named parameter tree from a flat vector."""
function unpack_parameters(spec::ContinuousHopfield, packed::AbstractVector)
    sizes = spec.layer_sizes
    offset = 0
    take_matrix(rows, columns) = begin
        count = rows * columns
        result = reshape(copy(view(packed, (offset + 1):(offset + count))), rows, columns)
        offset += count
        result
    end
    take_vector(width) = begin
        result = copy(view(packed, (offset + 1):(offset + width)))
        offset += width
        result
    end

    expected = sum(sizes[i + 1] * sizes[i] for i in 1:(length(sizes) - 1))
    spec.bias && (expected += sum(_dynamic_sizes(spec)))
    spec.recurrent && (expected += sum(size^2 for size in _dynamic_sizes(spec)))
    length(packed) == expected || throw(DimensionMismatch(
        "packed parameters have length $(length(packed)); expected $expected",
    ))

    weights = ntuple(i -> take_matrix(sizes[i + 1], sizes[i]), length(sizes) - 1)
    biases = spec.bias ? ntuple(i -> take_vector(sizes[i + 1]), length(sizes) - 1) : ()
    recurrent = spec.recurrent ?
        ntuple(i -> take_matrix(sizes[i + 1], sizes[i + 1]), length(sizes) - 1) : ()
    return (weights=weights, biases=biases, recurrent=recurrent)
end
