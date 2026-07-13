module EquilibriumPropagationOrdinaryDiffEqRosenbrockExt

using EquilibriumPropagation
import OrdinaryDiffEqRosenbrock
import SciMLBase

const SupportedODEAlgorithm = OrdinaryDiffEqRosenbrock.Rodas5P

include("ordinarydiffeq_relaxation.jl")

end
