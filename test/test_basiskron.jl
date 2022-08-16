using ContinuumArrays, QuasiArrays, StaticArrays, Test

@testset "Basis Kron" begin
    L = LinearSpline(range(0,1; length=4))
    K = QuasiKron(L, L)

    xy = axes(K,1)
    f = xy -> ((x,y) = xy; exp(x*cos(y)))
    K \ f.(xy)

end