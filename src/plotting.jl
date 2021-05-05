

_mul_plotgrid(_, args) = grid(first(args))
_mul_plotgrid(::Tuple{Any,PaddedLayout}, (P,c)) = plotgrid(P[:,colsupport(c)])

function _plotgrid(lay::ApplyLayout{typeof(*)}, P)
    args = arguments(lay,P)
    _mul_plotgrid(map(MemoryLayout,args), args)
end

_plotgrid(_, P) = grid(P)

plotgrid(g) = _plotgrid(MemoryLayout(g), g)

@recipe function f(g::AbstractQuasiVector)
    x = plotgrid(g)
    x,g[x]
end

@recipe function f(g::AbstractQuasiMatrix)
    x = plotgrid(g)
    x,g[x,:]
end