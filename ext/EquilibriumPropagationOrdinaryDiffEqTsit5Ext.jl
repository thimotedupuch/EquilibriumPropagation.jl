module EquilibriumPropagationOrdinaryDiffEqTsit5Ext

using EquilibriumPropagation
import OrdinaryDiffEqTsit5
import SciMLBase

const SupportedODEAlgorithm = OrdinaryDiffEqTsit5.Tsit5

include("ordinarydiffeq_relaxation.jl")

end
