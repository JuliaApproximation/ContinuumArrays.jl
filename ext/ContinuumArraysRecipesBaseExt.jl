module ContinuumArraysRecipesBaseExt

using ContinuumArrays, RecipesBase
using ContinuumArrays: plotgridvalues, splitdiracs, AbstractQuasiArray, ApplyQuasiVector, _split_svec

@recipe function f(g::AbstractQuasiArray)
    rδ = splitdiracs(g)
    if rδ === nothing
        x,v = plotgridvalues(g)
        tuple(_split_svec(x)..., v)
    else
        # plot the deltas as a secondary series so they share colour and legend entry
        r,δ = rδ
        @series begin
            r
        end
        primary := false
        δ
    end
end

# draw deltas as arrows
@recipe function f(δ::Union{DiracDelta, ApplyQuasiVector{<:Any,typeof(*),<:Tuple{Spline{-1},Any}}})
    arrow --> true
    plotgridvalues(δ)
end

end # module