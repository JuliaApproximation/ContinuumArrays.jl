using ContinuumArrays, QuasiArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, ForwardDiff, Test
import ContinuumArrays: ℵ₁, materialize, SimplifyStyle, AffineQuasiVector, BasisLayout, AdjointBasisLayout, SubBasisLayout, MappedBasisLayout
import QuasiArrays: SubQuasiArray, MulQuasiMatrix, Vec, Inclusion, QuasiDiagonal, LazyQuasiArrayApplyStyle, LazyQuasiArrayStyle
import LazyArrays: MemoryLayout, ApplyStyle, Applied, colsupport, arguments, ApplyLayout
import ForwardDiff: Dual


@testset "Inclusion" begin
    x = Inclusion(-1..1)
    @test x[0.1] === 0.1
    @test x[0.0] === 0.0
    x = Inclusion(-1.0..1)
    X = QuasiDiagonal(x)
    @test X[-1:0.1:1,-1:0.1:1] == Diagonal(-1:0.1:1)
    @test Base.BroadcastStyle(typeof(x)) == LazyQuasiArrayStyle{1}()
    @test x .* x isa BroadcastQuasiArray
    @test (x.*x)[0.1] == 0.1^2
end

@testset "DiracDelta" begin
    δ = DiracDelta(-1..3)
    @test axes(δ) === (axes(δ,1),) === (Inclusion(-1..3),)
    @test size(δ) === (length(δ),) === (ℵ₁,)
    @test δ[1.1] === 0.0
    @test δ[0.0] === Inf
    @test Base.IndexStyle(δ) === Base.IndexLinear()
end

@testset "HeavisideSpline" begin
    H = HeavisideSpline([1,2,3])

    @test axes(H) === (axes(H,1),axes(H,2)) === (Inclusion(1..3), Base.OneTo(2))
    @test size(H) === (size(H,1),size(H,2)) === (ℵ₁, 2)

    @test_throws BoundsError H[0.1, 1]
    @test H[1.1,1] === H'[1,1.1] === transpose(H)[1,1.1] === 1.0
    @test H[2.1,1] === H'[1,2.1] === transpose(H)[1,2.1] === 0.0
    @test H[1.1,2] === 0.0
    @test H[2.1,2] === 1.0
    @test_throws BoundsError H[2.1,3]
    @test_throws BoundsError H'[3,2.1]
    @test_throws BoundsError transpose(H)[3,2.1]
    @test_throws BoundsError H[3.1,2]

    @test all(H[[1.1,2.1], 1] .=== H'[1,[1.1,2.1]] .=== transpose(H)[1,[1.1,2.1]] .=== [1.0,0.0])
    @test all(H[1.1,1:2] .=== H[1.1,:] .=== [1.0,0.0])
    @test all(H[[1.1,2.1], 1:2] .=== [1.0 0.0; 0.0 1.0])

    @test_throws BoundsError H[[0.1,2.1], 1]
    @test MemoryLayout(typeof(H)) == BasisLayout()
    @test ApplyStyle(*, typeof(H), typeof([1,2])) isa LazyQuasiArrayApplyStyle
    f = H*[1,2]
    @test axes(f) == (Inclusion(1.0..3.0),)
    @test f[1.1] ≈ 1
    @test f[2.1] ≈ 2

    @test ApplyStyle(*,typeof(H'),typeof(H)) == SimplifyStyle()

    @test H'H == materialize(applied(*,H',H)) == Eye(2)
end

@testset "LinearSpline" begin
    L = LinearSpline([1,2,3])
    @test size(L) == (ℵ₁, 3)

    @test_throws BoundsError L[0.1, 1]
    @test L[1.1,1] == L'[1,1.1] == transpose(L)[1,1.1] ≈ 0.9
    @test L[2.1,1] === L'[1,2.1] === transpose(L)[1,2.1] === 0.0
    @test L[1.1,2] ≈ 0.1
    @test L[2.1,2] ≈ 0.9
    @test L[2.1,3] == L'[3,2.1] == transpose(L)[3,2.1] ≈ 0.1
    @test_throws BoundsError L[3.1,2]

    @test L[[1.1,2.1], 1] == L'[1,[1.1,2.1]] == transpose(L)[1,[1.1,2.1]] ≈ [0.9,0.0]
    @test L[1.1,1:2] ≈ [0.9,0.1]
    @test L[[1.1,2.1], 1:2] ≈ [0.9 0.1; 0.0 0.9]

    @test_throws BoundsError L[[0.1,2.1], 1]

    f = L*[1,2,4]
    @test axes(f) == (Inclusion(1.0..3.0),)
    @test f[1.1] ≈ 1.1
    @test f[2.1] ≈ 2.2

    δ = DiracDelta(1.2,1..3)
    L = LinearSpline([1,2,3])
    @test δ'L ≈ [0.8, 0.2, 0.0]

    @test L'L == SymTridiagonal([1/3,2/3,1/3], [1/6,1/6])

    @testset "==" begin
        L = LinearSpline([1,2,3])
        H = HeavisideSpline([1,2,3])
        @test L == L
        @test L ≠ H
        H = HeavisideSpline([1,1.5,2.5,3])
        @test_throws ArgumentError L == H
    end

    @testset "Adjoint layout" begin
        L = LinearSpline([1,2,3])
        @test MemoryLayout(typeof(L')) == AdjointBasisLayout()
        @test [3,4,5]'*L' isa ApplyQuasiArray
    end

    @testset "Broadcast layout" begin
        L = LinearSpline([1,2,3])
        b = BroadcastQuasiArray(+, L*[3,4,5], L*[1.,2,3])
        @test (L\b) == [4,6,8]
        B = BroadcastQuasiArray(+, L, L)
        @test L\B == 2Eye(3)

        b = BroadcastQuasiArray(-, L*[3,4,5], L*[1.,2,3])
        @test (L\b) == [2,2,2]
        B = BroadcastQuasiArray(-, L, L)
        @test L\B == 0Eye(3)
    end
end

@testset "Derivative" begin
    L = LinearSpline([1,2,3])
    f = L*[1,2,4]
    @test f[1.2] == 1.2

    D = Derivative(axes(L,1))
    @test ApplyStyle(*,typeof(D),typeof(L)) isa SimplifyStyle
    @test D*L isa MulQuasiMatrix
    @test length((D*L).args) == 2
    @test eltype(D*L) == Float64

    M = applied(*, (D*L).args..., [1,2,4])
    @test M isa Applied{LazyQuasiArrayApplyStyle}
    @test eltype(materialize(M)) == Float64

    M = applied(*, D, L, [1,2,4])
    @test M isa Applied{LazyQuasiArrayApplyStyle}

    fp = D*L*[1,2,4]

    @test eltype(fp) == Float64

    @test fp isa Vec
    @test length(fp.args) == 2
    @test fp[1.1] ≈ 1
    @test fp[2.2] ≈ 2


    fp = D*f
    @test length(fp.args) == 2
    @test fp[1.1] ≈ 1
    @test fp[2.2] ≈ 2
end

@testset "Weak Laplacian" begin
    H = HeavisideSpline(0:2)
    L = LinearSpline(0:2)
    D = Derivative(axes(L,1))

    M = QuasiArrays.flatten(Mul(D',D*L))
    @test length(M.args) == 3
    @test last(M.args) isa BandedMatrix

    @test ApplyStyle(*, typeof(L'), typeof(D')) == SimplifyStyle()
    @test apply(*,L',D') isa MulQuasiMatrix
    @test MemoryLayout(typeof(L')) isa AdjointBasisLayout
    @test (L'D') isa MulQuasiMatrix

    A = (L'D') * (D*L)
    @test A isa BandedMatrix
    @test A == (D*L)'*(D*L) == [1.0 -1 0; -1.0 2.0 -1.0; 0.0 -1.0 1.0]
    @test bandwidths(A) == (1,1)
end

@testset "Views" begin
    L = LinearSpline(0:2)
    @test view(L,0.1,1)[1] == L[0.1,1]

    L = LinearSpline(0:2)
    B1 = view(L,:,1)
    @test B1 isa SubQuasiArray{Float64,1}
    @test size(B1) == (ℵ₁,)
    @test B1[0.1] == L[0.1,1]
    @test_throws BoundsError B1[2.2]

    B = view(L,:,1:2)
    @test B isa SubQuasiArray{Float64,2}
    @test B[0.1,:] == L[0.1,1:2]

    B = @view L[:,2:end-1]
    @test B[0.1,:] == [0.1]

    L = LinearSpline([1,2,3,4])
    @test L[:,2:3] isa SubQuasiArray
    @test axes(L[:,2:3]) ≡ (Inclusion(1..4), Base.OneTo(2))
    @test L[:,2:3][1.1,1] == L[1.1,2]
    @test_throws BoundsError L[0.1,1]
    @test_throws BoundsError L[1.1,0]

    @test MemoryLayout(typeof(L[:,2:3])) isa SubBasisLayout
    @test L\L[:,2:3] isa BandedMatrix
    @test L\L[:,2:3] == [0 0; 1 0; 0 1.0; 0 0]

    @testset "Subindex of splines" begin
        L = LinearSpline(range(0,stop=1,length=10))
        @test L[:,2:end-1] isa SubQuasiArray
        @test L[:,2:end-1][0.1,1] == L[0.1,2]
        v = randn(8)
        f = L[:,2:end-1] * v
        @test f[0.1] ≈ (L*[0; v; 0])[0.1]
    end
end

@testset "Poisson" begin
    L = LinearSpline(range(0,stop=1,length=10))
    B = L[:,2:end-1] # Zero dirichlet by dropping first and last spline
    @test (B'B) == (L'L)[2:end-1,2:end-1]
    D = Derivative(axes(L,1))
    @test apply(*,D,B) isa SubQuasiArray
    @test D*B isa MulQuasiMatrix
    @test apply(*,D,B)[0.1,1] == (D*B)[0.1,1] == 9
    @test apply(*,D,B)[0.2,1] == (D*B)[0.2,1] == -9
    Δ = -((D*B)'*(D*B)) # Weak Laplacian
    @test Δ isa BandedMatrix

    @test B'D' isa MulQuasiMatrix
    @test length((B'D').args) == 2

    @test Δ == -(*(B',D',D,B))
    @test Δ == -(B'D'D*B)
    @test Δ == -((B'D')*(D*B))
    @test_broken Δ == -B'*(D'D)*B
    @test Δ == -(B'*(D'D)*B)

    f = L*exp.(L.points) # project exp(x)
    u = B * (Δ \ (B'f))

    @test u[0.1] ≈ -0.06612902692412974
end

@testset "Helmholtz" begin
    L = LinearSpline(range(0,stop=1,length=10))
    B = L[:,2:end-1] # Zero dirichlet by dropping first and last spline
    D = Derivative(axes(L,1))

    A = -((B'D')*(D*B)) + 100^2*B'B # Weak Laplacian

    f = L*exp.(L.points) # project exp(x)
    u = B * (A \ (B'f))

    @test u[0.1] ≈ 0.00012678835289369413
end

@testset "AffineQuasiVector" begin
    x = Inclusion(0..1)
    @test 2x isa AffineQuasiVector
    @test (2x)[0.1] == 0.2
    @test_throws BoundsError (2x)[2]
    y = 2x .- 1
    @test y isa AffineQuasiVector
    @test y[0.1] == 2*(0.1)-1
    @test y/2 isa AffineQuasiVector
    @test 2\y isa AffineQuasiVector
    @test (y/2)[0.1] == (2\y)[0.1] == -0.4
    @test y .+ 1 isa AffineQuasiVector
    @test (y .+ 1)[0.1] == (1 .+ y)[0.1]
    @test (y .- 1)[0.1] == y[0.1]-1
    @test (1 .- y)[0.1] == 1-y[0.1]

    @test findfirst(isequal(0.1),x) == findlast(isequal(0.1),x) == 0.1
    @test findall(isequal(0.1), x) == [0.1]
    @test findfirst(isequal(2),x) == findlast(isequal(2),x) == nothing
    @test findall(isequal(2), x) == Float64[]

    @test findfirst(isequal(0.2),y) == findlast(isequal(0.2),y) == 0.6
    @test findfirst(isequal(2.3),y) == findlast(isequal(2.3),y) == nothing
    @test findall(isequal(0.2),y) == [0.6]
    @test findall(isequal(2),y) == Float64[]

    @test AffineQuasiVector(x)[0.1] == 0.1
    @test minimum(y) == -1
    @test maximum(y) == 1
    @test union(y) == Inclusion(-1..1)
    @test ContinuumArrays.inbounds_getindex(y,0.1) == y[0.1]
    @test ContinuumArrays.inbounds_getindex(y,2.1) == 2*2.1 - 1

    z = 1 .- x
    @test minimum(z) == 0.0
    @test maximum(z) == 1.0
    @test union(z) == Inclusion(0..1)

    @test !isempty(z)
    @test z == z
end

@testset "Change-of-variables" begin
    x = Inclusion(0..1)
    y = 2x .- 1
    L = LinearSpline(range(-1,stop=1,length=10))
    @test L[y,:][0.1,:] == L[2*0.1-1,:]

    D = Derivative(axes(L,1))
    H = HeavisideSpline(L.points)
    @test H\((D*L) * 2) ≈ (H\(D*L))*2 ≈ diagm(0 => fill(-9,9), 1 => fill(9,9))[1:end-1,:]

    @test MemoryLayout(typeof(L[y,:])) isa MappedBasisLayout
    a,b = arguments((D*L)[y,:])
    @test H[y,:]\a == Eye(9)
    @test H[y,:] \ (D*L)[y,:] isa BandedMatrix

    D = Derivative(x)
    @test (D*L[y,:])[0.1,1] ≈ -9
    @test H[y,:] \ (D*L[y,:]) isa BandedMatrix
    @test H[y,:] \ (D*L[y,:]) ≈ diagm(0 => fill(-9,9), 1 => fill(9,9))[1:end-1,:]

    B = L[y,2:end-1]
    @test MemoryLayout(typeof(B)) isa MappedBasisLayout
    @test B[0.1,1] == L[2*0.1-1,2]
    @test B\B == Eye(8)
    @test L[y,:] \ B == Eye(10)[:,2:end-1]
end
