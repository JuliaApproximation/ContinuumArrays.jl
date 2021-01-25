using ContinuumArrays, Test

import ContinuumArrays: PiecewiseBlockDiagonal, VcatBlockDiagonal

@testset "PiecewiseBlockDiagonal" begin
    S1 = LinearSpline(0:1)
    S2 = LinearSpline(2:3)
    S = PiecewiseBlockDiagonal(S1, S2)

    @test S[0.5,1:4] == [S1[0.5,1:2]; zeros(2)]
    @test S[2.5,1:4] == [zeros(2); S2[2.5,1:2]]
    @test_throws BoundsError S[1.5,2]
    @test_throws BoundsError S[0.5,5]
end

@testset "VcatBlockDiagonal" begin
    S1 = LinearSpline(0:1)
    S2 = LinearSpline(0:0.5:1)
    S = VcatBlockDiagonal(S1, S2)
    @test size(S,2) == 5
    @test S[0.1,1:5] == [vcat.(S1[0.1,:],0); vcat.(0, S2[0.1,:])]
    @test_throws BoundsError S[1.1,1]
    @test_throws BoundsError S[0.1,6]
end