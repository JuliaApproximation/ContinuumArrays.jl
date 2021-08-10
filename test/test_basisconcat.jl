using ContinuumArrays, BlockArrays, Test
import ContinuumArrays: PiecewiseBasis, VcatBasis, HvcatBasis, arguments, ApplyLayout, checkpoints, UnionDomain

@testset "ConcatBasis" begin
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
            @test axes(Sv,2) isa BlockedUnitRange
            @test Sv[0.5,1:4] == S[0.5,1:4]
        end

        @testset "UnionDomain with point checkpoints" begin
            @test 0 ∈ checkpoints(UnionDomain(0, 1..2))
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
        @test S[0.1,Block(1)] == [[S1[0.1,1] 0; 0 0], [S1[0.1,2] 0; 0 0]]
        @test S[0.1,Block(1)[1]] == [S1[0.1,1] 0; 0 0]
        @test S[0.1,getindex.(Block(1),1:2)] == [[S1[0.1,1] 0; 0 0], [S1[0.1,2] 0; 0 0]]
        D = Derivative(axes(S,1))
        @test_broken (D*S)[0.1,1] # throws error

        v = view(S, :, Block.(2:3))
        @test v[0.1,1] == S[0.1,3]
        @test blockisequal(axes(arguments(ApplyLayout{typeof(*)}(), v)[2],1), axes(S,2))
    end
end