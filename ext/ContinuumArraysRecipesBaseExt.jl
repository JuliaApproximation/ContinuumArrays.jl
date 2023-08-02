module ContinuumArraysRecipesBaseExt

using ContinuumArrays, RecipesBase
using ContinuumArrays: plotgridvalues, AbstractQuasiArray, _split_svec

@recipe function f(g::AbstractQuasiArray)
    x,v = plotgridvalues(g)
    tuple(_split_svec(x)..., v)
end

end # module