using ADTypes: AutoForwardDiff, NoAutoDiff
using CommonSolve
using EquilibriumPropagation
using ForwardDiff
using LinearAlgebra
using Optimisers
using Random
using Test

include("test_model_interface.jl")
include("test_hopfield.jl")
include("test_relaxation.jl")
if all(
    package -> Base.find_package(package) !== nothing,
    (
        "SciMLBase",
        "OrdinaryDiffEqTsit5",
        "OrdinaryDiffEqRosenbrock",
        "SteadyStateDiffEq",
        "NonlinearSolve",
    ),
)
    using NonlinearSolve
    using OrdinaryDiffEqRosenbrock
    using OrdinaryDiffEqTsit5
    using SciMLBase
    using SteadyStateDiffEq
    include("test_sciml_extensions.jl")
end
include("test_gradients.jl")
include("test_continuous.jl")
include("test_nonconservative.jl")
include("test_holomorphic.jl")
if Base.find_package("Lux") !== nothing
    using Lux
    include("test_lux_extension.jl")
    include("test_asymep_mlp_example.jl")
end
include("test_show.jl")
include("test_optimisers.jl")
include("test_examples.jl")
if Base.find_package("MLDatasets") !== nothing
    using MLDatasets
    include("test_mnist_example.jl")
end
