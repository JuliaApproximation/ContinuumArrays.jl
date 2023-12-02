using Documenter, ContinuumArrays

makedocs(
    modules = [ContinuumArrays],
    sitename="ContinuumArrays.jl",
    pages = Any[
        "Home" => "index.md"])

deploydocs(
    repo = "github.com/JuliaApproximation/ContinuumArrays.jl.git",
    push_preview = true
)
