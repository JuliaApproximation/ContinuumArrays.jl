plotgrid(g) = grid(g)

@recipe function f(g::AbstractQuasiVector)
    x = plotgrid(g)
    g[x]
end

@recipe function f(g::AbstractQuasiMatrix)
    x = plotgrid(g)
    g[x,:]
end