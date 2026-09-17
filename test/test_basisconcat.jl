using ContinuumArrays, BlockArrays, InfiniteArrays, StaticArrays, FillArrays, LazyArrays, Test
import ContinuumArrays: PiecewiseBasis, VcatBasis, HvcatBasis, arguments, ApplyLayout, checkpoints, UnionDomain,
                        Basis, basis, coefficients, ExpansionLayout, uplus_size
import ArrayLayouts: MemoryLayout
import InfiniteArrays: OneToInf

struct InfPolynomial{T,D} <: Basis{T}
    domain::D
end

InfPolynomial(d) = InfPolynomial{Float64,typeof(d)}(d)
Base.axes(P::InfPolynomial) = (Inclusion(P.domain), Base.oneto(∞))
Base.:(==)(P::InfPolynomial, Q::InfPolynomial) = P.domain == Q.domain
Base.getindex(P::InfPolynomial, x::Number, k::Int) = x^(k-1)

# ClassicalOrthogonalPolynomials.jl overloads this to use PiecewiseInterlace
uplus_size(ax::Tuple{Vararg{InfiniteCardinal{0}}}, Ps::Tuple, cs::Tuple) = (ax, Ps, cs)

@testset "ConcatBasis" begin
    @testset "hcat" begin
        x = Inclusion(-1..1)
        H = [ones(x) x]
        @test H[:,1:2] isa SubQuasiArray
        Q  =QuasiFill(2,(x,Base.oneto(∞)))
        H∞ = [ones(x) Q]
        @test H∞[:,1:∞] isa SubQuasiArray
        @test (H∞')[1:∞,:] isa SubQuasiArray

        𝐱 = Inclusion((-1.0..1)^2)
        @test 𝐱[SVector(0.1,0.2)] == SVector(0.1,0.2)
        H = [first.(𝐱) last.(𝐱)]
        @test H[SVector(0.1,0.2),1] == 0.1
        @test H[SVector(0.1,0.2),1:2] == H[SVector(0.1,0.2),:] == [0.1, 0.2]
        @test H[[SVector(0.1,0.2),SVector(0.3,0.4)],1] == [0.1,0.3]
        @test H[[SVector(0.1,0.2),SVector(0.3,0.4)],1:2] == H[[SVector(0.1,0.2),SVector(0.3,0.4)],:] == [0.1 0.2; 0.3 0.4]
    end
    @testset "PiecewiseBasis" begin
        S1 = LinearSpline(0:1)
        S2 = LinearSpline(2:3)
        S = PiecewiseBasis(S1, S2)

        @test S == S == copy(S)
        @test checkpoints(S) == union(checkpoints(S1), checkpoints(S2))

        @test S[0.5,1:4] == [S1[0.5,1:2]; zeros(2)]
        @test S[2.5,1:4] == [zeros(2); S2[2.5,1:2]]
        @test_throws BoundsError S[1.5,2]
        @test_throws BoundsError S[0.5,5]

        D = Derivative(axes(S,1))
        D1 = Derivative(axes(S1,1))
        D2 = Derivative(axes(S2,1))
        @test (D*S)[0.5,1:4] == [(D1 * S1)[0.5,1:2]; zeros(2)]
        @test (D*S)[2.5,1:4] == [zeros(2); (D2 * S2)[2.5,1:2]]

        @test_throws BoundsError (D*S)[1.5,2]
        @test_throws BoundsError (D*S)[0.5,5]

        @test_throws DimensionMismatch D1*S

        @testset "Vec case" begin
            Sv = PiecewiseBasis([S1,S2])
            @test axes(Sv,2) isa BlockedOneTo
            @test Sv[0.5,1:4] == S[0.5,1:4]
            @test  Sv[0.5,Block(1)] == [0.5,0.5]
        end

        @testset "UnionDomain with point checkpoints" begin
            @test 0 ∈ checkpoints(UnionDomain(0, 1..2))
        end
    end

    @testset "⊎" begin
        S1 = LinearSpline(0:1)
        S2 = LinearSpline(2:3)
        f = S1 * [1.,2.]
        g = S2 * [3.,4.]
        h = f ⊎ g

        @test MemoryLayout(h) isa ExpansionLayout
        @test basis(h) == PiecewiseBasis(S1, S2)
        @test coefficients(h) == [1,2,3,4]
        @test blockisequal(axes(coefficients(h),1), axes(basis(h),2))
        @test h[0.5] == f[0.5]
        @test h[2.5] == g[2.5]
        @test (h .+ h)[0.5] == 2f[0.5]
        @test basis(h) \ h == coefficients(h)

        S3 = LinearSpline(4:5)
        u = S3 * [5.,6.]
        h3 = ⊎(f, g, u)
        @test basis(h3) == PiecewiseBasis(S1, S2, S3)
        @test h3[4.5] == 5.5

        @testset "associativity" begin
            @test basis((f ⊎ g) ⊎ u) == basis(f ⊎ (g ⊎ u)) == basis(h3)
            @test coefficients((f ⊎ g) ⊎ u) == coefficients(f ⊎ (g ⊎ u)) == coefficients(h3)
            for x in (0.5, 2.5, 4.5)
                @test ((f ⊎ g) ⊎ u)[x] == (f ⊎ (g ⊎ u))[x] == h3[x]
            end

            # coefficients that are not blocked can still be split back into pieces
            m = PiecewiseBasis(S1, S2) * [1.,2.,3.,4.]
            @test basis(m ⊎ u) == basis(h3)
            @test coefficients(m ⊎ u) == coefficients(h3)
        end

        @testset "infinite axes" begin
            P1 = InfPolynomial(0..1)
            P2 = InfPolynomial(2..3)
            u = P1 * Vcat([1.,2.], Zeros(∞))
            v = P2 * Vcat([3.,4.], Zeros(∞))
            @test MemoryLayout(u) isa ExpansionLayout

            ax,Ps,cs = u ⊎ v
            @test ax == (Base.oneto(∞), Base.oneto(∞))
            @test Ps == (P1, P2)
            @test cs === (coefficients(u), coefficients(v))
        end
    end

    @testset "VcatBasis" begin
        S1 = LinearSpline(0:1)
        S2 = LinearSpline(0:0.5:1)
        S = VcatBasis(S1, S2)

        @test size(S,2) == 5
        @test axes(S,1) == axes(S1,1) == axes(S2,1)
        @test blockaxes(S) == (Block.(1:1), Block.(1:2))

        @test S == S

        @test S[0.1,1:5] == [vcat.(S1[0.1,:],0); vcat.(0, S2[0.1,:])]
        @test_throws BoundsError S[1.1,1]
        @test_throws BoundsError S[0.1,6]

        @test permutedims(S)[1:5,0.1] == S[0.1,1:5]

        D = Derivative(axes(S,1))
        @test (D*S)[0.1,1:5] == [vcat.((D*S1)[0.1,:],0); vcat.(0, (D*S2)[0.1,:])]
        
        @test_throws BoundsError (D*S)[1.5,2]
        @test_throws BoundsError (D*S)[0.5,6]

        H = VcatBasis(HeavisideSpline(S1.points), HeavisideSpline(S2.points))
        @test H \ (D*S) == [-1 1 0 0 0; 0 0 -2 2 0; 0 0 0 -2 2]
    end

    @testset "HvcatBasis" begin
        S1 = LinearSpline(0:1)
        S2 = LinearSpline(0:0.5:1)
        S = HvcatBasis(2, S1, S2, S2, S1)
        D = Derivative(axes(S,1))

        @test S == S

        @test S[0.1, 1] == [S1[0.1,1] 0; 0 0]
        @test S[0.1,Block(1)[1]] == [S1[0.1,1] 0; 0 0]
        @test S[0.1,[Block(1)[1]]] == S[0.1,Block(1)[1:1]] == S[0.1,[Block(1)[1:1]]] == [[S1[0.1,1] 0; 0 0]]
        @test S[0.1,Block.(1:1)] == S[0.1,Block(1)] == S[0.1,[Block(1)]] == [[S1[0.1,1] 0; 0 0], [S1[0.1,2] 0; 0 0]]
        @test S[0.1,getindex.(Block(1),1:2)] == [[S1[0.1,1] 0; 0 0], [S1[0.1,2] 0; 0 0]]
        D = Derivative(axes(S,1))
        @test_broken (D*S)[0.1,1] # throws error

        v = view(S, :, Block.(2:3))
        @test v[0.1,1] == S[0.1,3]
        @test blockisequal(axes(arguments(ApplyLayout{typeof(*)}(), v)[2],1), axes(S,2))
    end
end