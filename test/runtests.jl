using ContinuumArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, Test,
    InfiniteArrays
    import ContinuumArrays: ℵ₁, materialize
    import ContinuumArrays.QuasiArrays: SubQuasiArray, MulQuasiMatrix, Vec, Inclusion, QuasiDiagonal
    import LazyArrays: rmaterialize


@testset "Inclusion" begin
    @test Inclusion(-1..1)[0.0] === 0
    @test_throws InexactError Inclusion(-1..1)[0.1]
    X = QuasiDiagonal(Inclusion(-1.0..1))
    @test X[-1:0.1:1,-1:0.1:1] == Diagonal(-1:0.1:1)
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

    f = H*[1,2]
    @test axes(f) == (Inclusion(1.0..3.0),)
    @test f[1.1] ≈ 1
    @test f[2.1] ≈ 2

    @test H'H == Eye(2)
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
    L[[1.1,2.1], 1]
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
    D = Derivative(axes(L,1))
    @test D*L isa MulQuasiMatrix

    fp = (D*L)*[1,2,4]
    @test fp isa Vec
    @test length(fp.mul.args) == 2
    @test fp[1.1] ≈ 1
    @test fp[2.2] ≈ 2

    fp = D*f
    @test length(fp.mul.args) == 2

    @test fp[1.1] ≈ 1
    @test fp[2.2] ≈ 2
end

@testset "Weak Laplacian" begin
    H = HeavisideSpline(0:2)
    L = LinearSpline(0:2)

    D = Derivative(axes(L,1))
    M = rmaterialize(Mul(D',D*L))
    @test length(M.mul.args) == 3
    @test last(M.mul.args) isa BandedMatrix

    @test M.mul.args == rmaterialize(Mul(D',D,L)).mul.args ==
        *(D',D,L).mul.args

    @test (L'D') isa MulQuasiMatrix
    A = (L'D') * (D*L)
    @test A == (D*L)'*(D*L) == [1.0 -1 0; -1.0 2.0 -1.0; 0.0 -1.0 1.0]

    @test A isa MulArray
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
    @test_broken L[:,2:end-1][0.1,1] == L[0.1,2]
    v = randn(8)
    f = L[:,2:end-1] * v
    @test f[0.1] ≈ (L*[0; v; 0])[0.1]
end

@testset "Poisson" begin
    L = LinearSpline(range(0,stop=1,length=10))
    B = L[:,2:end-1] # Zero dirichlet by dropping first and last spline
    D = Derivative(axes(L,1))
    Δ = -((B'D')*(D*B)) # Weak Laplacian

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

@testset "Jacobi" begin
    S = Jacobi(true,true)
    W = Diagonal(JacobiWeight(true,true))
    D = Derivative(axes(W,1))
    P = Legendre()

    Bi = pinv(Jacobi(2,2))
    @test Bi isa ContinuumArrays.QuasiArrays.PInvQuasiMatrix
    @test PInv(P)*P === pinv(P)*P === Eye(∞)

    A = @inferred(PInv(Jacobi(2,2))*(D*S))
    @test typeof(A) == typeof(pinv(Jacobi(2,2))*(D*S))

    @test A isa MulMatrix
    @test isbanded(A)
    @test bandwidths(A) == (-1,1)
    @test size(A) == (∞,∞)
    @test A[1:10,1:10] == diagm(1 => 1:0.5:5)

    M = @inferred(D*S)
    @test M isa MulQuasiMatrix
    @test M.mul.args[1] == Jacobi(2,2)
    @test M.mul.args[2][1:10,1:10] == A[1:10,1:10]

    L = Diagonal(JacobiWeight(true,false))
    A = @inferred(pinv(Jacobi(false,true))*L*S)
    @test A isa BandedMatrix
    @test size(A) == (∞,∞)

    L = Diagonal(JacobiWeight(false,true))
    A = @inferred(pinv(Jacobi(true,false))*L*S)
    @test A isa BandedMatrix
    @test size(A) == (∞,∞)

    A,B = (P'P),(pinv(P)*W*S)

    M = Mul(A,B)
    @test M[1,1] == 4/3

    M = MulMatrix{Float64}(A,B)
    M̃ = M[1:10,1:10]
    @test M̃ isa BandedMatrix
    @test bandwidths(M̃) == (2,0)

    @test A*B isa MulArray

    A,B,C = (pinv(P)*W*S)',(P'P),(pinv(P)*W*S)
    M = MulArray(A,B,C)
    @test typeof(A*B*C) == typeof(M)
    @test M[1,1] ≈  1+1/15
end

@testset "P-FEM" begin
    S = Jacobi(true,true)
    W = Diagonal(JacobiWeight(true,true))
    D = Derivative(axes(W,1))
    P = Legendre()
    N = 10

    @test fullmaterialize(pinv(P)*(D*W)*S[:,1:N]) isa AbstractMatrix

    L = fullmaterialize(D*W*S[:,1:N])
    Δ = L'L
    @test Δ isa MulMatrix
    @test bandwidths(Δ) == (0,0)

    L = (D*W*S[:,1:N])
    Δ = fullmaterialize(L'L)
    @test Δ isa Matrix
end
