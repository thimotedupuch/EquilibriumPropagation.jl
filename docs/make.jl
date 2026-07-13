using Documenter
using EquilibriumPropagation

makedocs(
    sitename="EquilibriumPropagation.jl",
    modules=[EquilibriumPropagation],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical=nothing,
        edit_link=nothing,
        repolink=nothing,
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "EP concepts" => "concepts.md",
        "Optional equilibrium solvers" => "solvers.md",
        "Continuous EP" => "continuous_ep.md",
        "Non-conservative EP" => "nonconservative_ep.md",
        "Holomorphic EP" => "holomorphic_ep.md",
        "Lux integration" => "lux.md",
        "OpenXLA acceleration with Reactant" => "reactant.md",
        "Continuous Hopfield example" => "hopfield_spiral.md",
        "MNIST classification example" => "mnist.md",
        "Feedforward AsymEP MLP" => "asymep_mlp.md",
        "API reference" => "api.md",
    ],
    checkdocs=:exports,
    remotes=nothing,
    warnonly=false,
)
