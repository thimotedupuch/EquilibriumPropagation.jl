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
        "Continuous Hopfield example" => "hopfield_spiral.md",
        "MNIST classification example" => "mnist.md",
        "API reference" => "api.md",
    ],
    checkdocs=:exports,
    remotes=nothing,
    warnonly=false,
)
