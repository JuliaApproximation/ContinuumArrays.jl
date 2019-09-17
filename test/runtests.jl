using ContinuumArrays, QuasiArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, Test, ForwardDiff
import ContinuumArrays: ℵ₁, materialize, SimplifyStyle
import QuasiArrays: SubQuasiArray, MulQuasiMatrix, Vec, Inclusion, QuasiDiagonal, LazyQuasiArrayApplyStyle, LazyQuasiArrayStyle, LmaterializeApplyStyle
import LazyArrays: MemoryLayout, ApplyStyle, Applied, colsupport
import ForwardDiff: Dual


@testset "Inclusion" begin
    x = Inclusion(-1..1)
    @test_throws InexactError x[0.1]
    @test x[0.0] === 0
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

    @test axes(H) === (axes(H,1),axes(H,2)) === (Inclusion(1.0..3.0), Base.OneTo(2))
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
end

@testset "Subindex of splines" begin
    L = LinearSpline(range(0,stop=1,length=10))
    @test L[:,2:end-1] isa MulQuasiMatrix
    @test L[:,2:end-1][0.1,1] == L[0.1,2]
    v = randn(8)
    f = L[:,2:end-1] * v
    @test f[0.1] ≈ (L*[0; v; 0])[0.1]
end

@testset "Poisson" begin
    L = LinearSpline(range(0,stop=1,length=10))
    B = L[:,2:end-1] # Zero dirichlet by dropping first and last spline
    D = Derivative(axes(L,1))
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

