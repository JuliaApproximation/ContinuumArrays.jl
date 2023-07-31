module ContinuumArraysRecipesBaseExt

using ContinuumArrays, RecipesBase, StaticArrays
using ContinuumArrays: plotgridvalues, AbstractQuasiArray

_split_svec(x) = (x,)
_split_svec(x::AbstractArray{<:StaticVector{2}}) = (map(first,x), map(last,x))


@recipe function f(g::AbstractQuasiArray)
    x,v = plotgridvalues(g)
    tuple(_split_svec(x)..., v)
end

end # module