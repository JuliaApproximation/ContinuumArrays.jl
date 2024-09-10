
const MAX_PLOT_POINTS = 10_000 # above this rendering is too slow


"""
    plotgrid(P, n...)

returns a grid of points suitable for plotting. This may include
endpoints or singular points not included in `grid`. `n` specifies
the number of coefficients.
"""
plotgrid(P, n...) = plotgrid_layout(MemoryLayout(P), P, n...)

plotgrid_layout(lay, P, n...) = plotgrid_size(size(P), P, n...)
plotgrid_size(::Tuple{InfiniteCardinal{1}}, P, n...) = plotgrid(expand(P), n...)
plotgrid_size(sz, P, n=size(P,2)) = grid(P, min(n,MAX_PLOT_POINTS))
plotgrid_layout(::WeightedBasisLayouts, wP, n...) = plotgrid(unweighted(wP), n...)
plotgrid_layout(::MappedBasisLayout, P, n...) = invmap(parentindices(P)[1])[plotgrid(demap(P), n...)]
plotgrid_layout(::SubBasisLayout, P::AbstractQuasiMatrix, n) = plotgrid(parent(P), maximum(parentindices(P)[2][n]))
plotgrid_layout(::SubBasisLayout, P::AbstractQuasiMatrix) = plotgrid(parent(P), maximum(parentindices(P)[2]))


_mul_plotgrid(_, args) = plotgrid(args[1], last(colsupport(ApplyArray(*, tail(args)...))))
_mul_plotgrid(_, (P,c)::NTuple{2,Any}) = plotgrid(P, last(colsupport(c)))

function plotgrid_layout(lay::ApplyLayout{typeof(*)}, P)
    args = arguments(lay,P)
    _mul_plotgrid(map(MemoryLayout,args), args)
end

plotgrid_layout(::ExpansionLayout, P) = plotgrid_layout(ApplyLayout{typeof(*)}(), P)

plotvalues_size(::Tuple{InfiniteCardinal{1}}, g, x=plotgrid(g)) = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1},Int}, g, x=plotgrid(g)) = g[x,:]
plotvalues_layout(lay, g, x...) = plotvalues_size(size(g), g, x...)
# plotvalues_layout(::WeightedBasisLayouts, wP, n...) = plotvalues(unweighted(wP), n...)
plotvalues_layout(::ExpansionLayout{MappedBasisLayout}, g, x...) = plotvalues(demap(g))
function plotvalues_layout(::ExpansionLayout{<:WeightedBasisLayout}, g, x...)
    f = unweighted(g)
    w = weight(basis(g))
    x = plotgrid(g)
    w[x] .* plotvalues(f)
end
# plotvalues_layout(::SubBasisLayout, P::AbstractQuasiMatrix, n) = plotvalues(parent(P), maximum(parentindices(P)[2][n]))
# plotvalues_layout(::SubBasisLayout, P::AbstractQuasiMatrix) = plotvalues(parent(P), maximum(parentindices(P)[2]))

plotvalues(g::AbstractQuasiArray, x...) = plotvalues_layout(MemoryLayout(g), g, x...)

function plotgridvalues(g)
    x = plotgrid(g)
    x, plotvalues(g,x)
end

_split_svec(x) = (x,)
_split_svec(x::AbstractArray{<:StaticVector{2}}) = (map(first,x), map(last,x))
