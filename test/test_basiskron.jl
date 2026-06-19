using ContinuumArrays, QuasiArrays, StaticArrays, Test
using ContinuumArrays: KronExpansionLayout

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

    @test F[:,[0.1,0.2]][0.3,:] ≈ F[0.3,[0.1,0.2]] ≈ F[0.3,:][[0.1,0.2]]
    @test F[[0.1,0.2],:][:,0.3] ≈ F[[0.1,0.2],0.3] ≈ F[:,0.3][[0.1,0.2]]
    @test F[[0.1,0.2],[0.3,0.4]] ≈ F[[0.1,0.2],:][:,[0.3,0.4]] ≈ F[:,[0.3,0.4]][[0.1,0.2],:]

    @test diff(F)[0.1,0.2] ≈ diff(F;dims=1)[0.1,0.2]  ≈ diff(L)[0.1,:]'*C*L[0.2,:]
    @test diff(F;dims=2)[0.1,0.2]  ≈ L[0.1,:]'*C*diff(L)[0.2,:]

    @test L\F/L' == (L\F)/L' == L\(F/L') == C

    @testset "real/imag" begin
        F = L * (C .+ im) * L'
        @test real(F)[0.1,0.2] == (L * C * L')[0.1,0.2]
        @test imag(F)[0.1,0.2] == (L * one.(C) * L')[0.1,0.2]
        @test MemoryLayout(real(F)) isa KronExpansionLayout 
        @test MemoryLayout(imag(F)) isa KronExpansionLayout
    end

    @testset "plot" begin
        F = L * C * L'
        ((x,y), Z) = ContinuumArrays.plotgridvalues(F)
        @test F[x,y] == Z
    end
end