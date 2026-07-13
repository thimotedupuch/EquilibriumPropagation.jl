using ADTypes: AutoForwardDiff, NoAutoDiff
using CommonSolve
using EquilibriumPropagation
using ForwardDiff
using LinearAlgebra
using Optimisers
using Test

include("test_model_interface.jl")
include("test_relaxation.jl")
include("test_gradients.jl")
include("test_show.jl")
include("test_optimisers.jl")
include("test_examples.jl")
