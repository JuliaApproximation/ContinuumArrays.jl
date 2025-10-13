using ContinuumArrays, QuasiArrays, StaticArrays, Test

@testset "Basis Kron" begin
    L = LinearSpline(range(0,1; length=4))
    K = QuasiKron(L, L)

    xy = axes(K,1)
    f = xy -> ((x,y) = xy; exp(x*cos(y)))
    @test_broken K \ f.(xy)
end

@testset "KronExpansion" begin
    L = LinearSpline(range(0,1; length=4))
    C = reshape(Vector(1:16), 4, 4)
    F = L * C * L'
    @test sum(F) ≈ 8.5
    @test sum(F; dims=1)[1,0.1] ≈ 3.7
    @test sum(F; dims=2)[0.1,1] ≈ 7.3

    @test F[0.1,:][0.2] ≈ F[:,0.2][0.1] ≈ F[0.1,0.2]
end