
const MAX_PLOT_POINTS = 10_000 # above this rendering is too slow


"""
    plotgrid(P, n...)

returns a grid of points suitable for plotting. This may include
endpoints or singular points not included in `grid`. `n` specifies
the number of coefficients.
"""

plotgrid(P, n...) = _plotgrid(MemoryLayout(P), P, n...)
_plotgrid(lay, P, n=size(P,2)) = grid(P, min(n,MAX_PLOT_POINTS))

_plotgrid(::WeightedBasisLayouts, wP, n...) = plotgrid(unweighted(wP), n...)
_plotgrid(::MappedBasisLayout, P, n...) = invmap(parentindices(P)[1])[plotgrid(demap(P), n...)]
_plotgrid(::SubBasisLayout, P::AbstractQuasiMatrix, n) = plotgrid(parent(P), maximum(parentindices(P)[2][n]))
_plotgrid(::SubBasisLayout, P::AbstractQuasiMatrix) = plotgrid(parent(P), maximum(parentindices(P)[2]))


_mul_plotgrid(_, args) = _plotgrid(UnknownLayout(), first(args))
_mul_plotgrid(::Tuple{Any,PaddedLayout}, (P,c)) = plotgrid(P, maximum(colsupport(c)))

function _plotgrid(lay::ExpansionLayout, P)
    args = arguments(lay,P)
    _mul_plotgrid(map(MemoryLayout,args), args)
end

_split_svec(x) = (x,)
_split_svec(x::AbstractArray{<:StaticVector{2}}) = (map(first,x), map(last,x))

plotvalues(g::AbstractQuasiVector, x) = g[x]
plotvalues(g::AbstractQuasiMatrix, x) = g[x,:]

function plotgridvalues(g)
    x = plotgrid(g)
    x, plotvalues(g,x)
end

@recipe function f(g::AbstractQuasiArray)
    x,v = plotgridvalues(g)
    tuple(_split_svec(x)..., v)
end
