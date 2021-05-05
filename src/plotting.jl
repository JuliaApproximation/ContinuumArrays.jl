

_mul_plotgrid(_, args) = grid(first(args))
_mul_plotgrid(::Tuple{Any,PaddedLayout}, (P,c)) = plotgrid(P[:,colsupport(c)])

function _plotgrid(lay::ApplyLayout{typeof(*)}, P)
    args = arguments(lay,P)
    _mul_plotgrid(map(MemoryLayout,args), args)
end

_plotgrid(_, P) = grid(P)

plotgrid(g) = _plotgrid(MemoryLayout(g), g)

_split_svec(x) = (x,)
_split_svec(x::AbstractArray{<:StaticVector{2}}) = (map(first,x), map(last,x))

@recipe function f(g::AbstractQuasiVector)
    x = plotgrid(g)
    tuple(_split_svec(x)...,g[x])
end

@recipe function f(g::AbstractQuasiMatrix)
    x = plotgrid(g)
    tuple(_split_svec(x)...,g[x,:])
end