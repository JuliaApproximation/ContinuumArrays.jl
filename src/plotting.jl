
const MAX_PLOT_POINTS = 10_000 # above this rendering is too slow

plotgrid(g) = _plotgrid(MemoryLayout(g), g)
_plotgrid(_, P) = grid(P, min(size(P,2),MAX_PLOT_POINTS))

_plotgrid(::WeightedBasisLayouts, wP) = plotgrid(unweighted(wP))
_plotgrid(::MappedBasisLayout, P) = invmap(parentindices(P)[1])[plotgrid(demap(P))]

_mul_plotgrid(_, args) = _plotgrid(UnknownLayout(), first(args))
_mul_plotgrid(::Tuple{Any,PaddedLayout}, (P,c)) = plotgrid(P[:,colsupport(c)])

function _plotgrid(lay::ExpansionLayout, P)
    args = arguments(lay,P)
    _mul_plotgrid(map(MemoryLayout,args), args)
end

_split_svec(x) = (x,)
_split_svec(x::AbstractArray{<:StaticVector{2}}) = (map(first,x), map(last,x))

plotvalues(g::AbstractQuasiVector, x) = g[x]
plotvalues(g::AbstractQuasiMatrix, x) = g[x,:]

@recipe function f(g::AbstractQuasiArray)
    x = plotgrid(g)
    tuple(_split_svec(x)..., plotvalues(g,x))
end
