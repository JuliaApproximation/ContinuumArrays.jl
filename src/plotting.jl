
const MAX_PLOT_POINTS = 10_000 # above this rendering is too slow


"""
    plotgrid(P, n...)

returns a grid of points suitable for plotting. This may include
endpoints or singular points not included in `grid`. `n` specifies
the number of coefficients.
"""

plotgrid(P, n...) = plotgrid_layout(MemoryLayout(P), P, n...)

plotgrid_layout(lay, P, n=size(P,2)) = grid(P, min(n,MAX_PLOT_POINTS))

plotgrid_layout(::WeightedBasisLayouts, wP, n...) = plotgrid(unweighted(wP), n...)
plotgrid_layout(::MappedBasisLayout, P, n...) = invmap(parentindices(P)[1])[plotgrid(demap(P), n...)]
plotgrid_layout(::SubBasisLayout, P::AbstractQuasiMatrix, n) = plotgrid(parent(P), maximum(parentindices(P)[2][n]))
plotgrid_layout(::SubBasisLayout, P::AbstractQuasiMatrix) = plotgrid(parent(P), maximum(parentindices(P)[2]))

const _plotgrid = plotgrid_layout # TODO: remove


_mul_plotgrid(_, args) = plotgrid_layout(UnknownLayout(), first(args))
_mul_plotgrid(::Tuple{Any,PaddedLayout}, (P,c)) = plotgrid(P, maximum(colsupport(c)))

function plotgrid_layout(lay::ExpansionLayout, P)
    args = arguments(lay,P)
    _mul_plotgrid(map(MemoryLayout,args), args)
end

plotvalues(g::AbstractQuasiVector, x) = g[x]
plotvalues(g::AbstractQuasiMatrix, x) = g[x,:]
plotvalues(g::AbstractQuasiArray) = plotvalues(g, plotgrid(g))

function plotgridvalues(g)
    x = plotgrid(g)
    x, plotvalues(g,x)
end

_split_svec(x) = (x,)
_split_svec(x::AbstractArray{<:StaticVector{2}}) = (map(first,x), map(last,x))
