using ContinuumArrays, Test
import ContinuumArrays: AffineQuasiVector

@testset "AffineQuasiVector" begin
    x = Inclusion(0..1)
    y = 2x .- 1
    a = affine(-2..3, -1..1)

    @testset "algebra" begin
        @test 2x isa AffineQuasiVector
        @test x/2 isa AffineQuasiVector
        @test (2x)[0.1] == 0.2
        @test_throws BoundsError (2x)[2]

        @test y isa AffineQuasiVector
        @test y[0.1] == 2*(0.1)-1
        @test y/2 isa AffineQuasiVector
        @test 2\y isa AffineQuasiVector
        @test (y/2)[0.1] == (2\y)[0.1] == -0.4
        @test y .+ 1 isa AffineQuasiVector
        @test (y .+ 1)[0.1] == (1 .+ y)[0.1]
        @test (y .- 1)[0.1] == y[0.1]-1
        @test (1 .- y)[0.1] == 1-y[0.1]
        @test y[x] == y[:] == y
        @test_throws BoundsError y[Inclusion(0..2)]

        @test y == y

        @test x == affine(x,x)
        @test affine(x,x) == x
    end

    @testset "find" begin
        @test findfirst(isequal(0.1),x) == findlast(isequal(0.1),x) == 0.1
        @test findall(isequal(0.1), x) == [0.1]
        @test findfirst(isequal(2),x) == findlast(isequal(2),x) == nothing
        @test findall(isequal(2), x) == Float64[]

        @test findfirst(isequal(0.2),y) == findlast(isequal(0.2),y) == 0.6
        @test findfirst(isequal(2.3),y) == findlast(isequal(2.3),y) == nothing
        @test findall(isequal(0.2),y) == [0.6]
        @test findall(isequal(2),y) == Float64[]
    end

    @testset "minmax" begin
        @test AffineQuasiVector(x)[0.1] == 0.1
        @test minimum(y) == -1
        @test maximum(y) == 1
        @test union(y) == Inclusion(-1..1)
        @test ContinuumArrays.inbounds_getindex(y,0.1) == y[0.1]
        @test ContinuumArrays.inbounds_getindex(y,2.1) == 2*2.1 - 1
        @test ContinuumArrays.inbounds_getindex(y, [2.1,2.2]) == 2*[2.1,2.2] .- 1

        z = 1 .- x
        @test minimum(z) == 0.0
        @test maximum(z) == 1.0
        @test union(z) == Inclusion(0..1)

        @test !isempty(z)
        @test z == z
    end

    @testset "AffineMap" begin
        @test a[0.1] == -0.16
        @test_throws BoundsError a[-3]
        @test a[-2] == first(a) == -1
        @test a[3] == last(a) == 1
        @test invmap(a)[-0.16] ≈ 0.1
        @test invmap(a)[1] == 3
        @test invmap(a)[-1] == -2
        @test union(a) == Inclusion(-1..1)

        @test affine(0..1, -1..1) == y
    end

    @testset "show" begin
        @test stringmime("text/plain", y) == "2.0 * Inclusion($(0..1)) .+ (-1.0)"
        @test stringmime("text/plain", a) == "Affine map from Inclusion($(-2..3)) to Inclusion($(-1..1))"
    end
end
