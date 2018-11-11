using ContinuumArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, Test,
    InfiniteArrays
    import ContinuumArrays: ℵ₁, materialize
    import ContinuumArrays.QuasiArrays: SubQuasiArray

@testset "DiracDelta" begin
    δ = DiracDelta(-1..3)
    @test axes(δ) === (axes(δ,1),) === (-1..3,)
    @test size(δ) === (length(δ),) === (ℵ₁,)
    @test δ[1.1] === 0.0
    @test δ[0.0] === Inf
    @test Base.IndexStyle(δ) === Base.IndexLinear()
end

@testset "HeavisideSpline" begin
    H = HeavisideSpline([1,2,3])
    @test axes(H) === (axes(H,1),axes(H,2)) === (1.0..3.0, Base.OneTo(2))
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
    @test axes(f) == (1.0..3.0,)
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
    @test axes(f) == (1.0..3.0,)
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
    fp = D*f

    @test fp[1.1] ≈ 1
    @test fp[2.2] ≈ 2
end

@testset "Weak Laplacian" begin
    H = HeavisideSpline(0:2)
    L = LinearSpline(0:2)

    D = Derivative(axes(L,1))
    M = materialize(Mul(D',D,L))
    DL = D*L
    @test M.factors == tuple(D', (D*L).factors...)

    @test materialize(Mul(L', D', D, L)) == (L'D'*D*L) ==
        [1.0 -1 0; -1.0 2.0 -1.0; 0.0 -1.0 1.0]

    @test materialize(Mul(L', D', D, L)) isa BandedMatrix
    @test (L'D'*D*L) isa BandedMatrix

    @test bandwidths(materialize(L'D'*D*L)) == (1,1)
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
    @test L[:,2:end-1] isa Mul
    @test_broken L[:,2:end-1][0.1,1] == L[0.1,2]
    v = randn(8)
    f = L[:,2:end-1] * v
    @test f[0.1] ≈ (L*[0; v; 0])[0.1]
end

@testset "Poisson" begin
    L = LinearSpline(range(0,stop=1,length=10))
    B = L[:,2:end-1] # Zero dirichlet by dropping first and last spline
    D = Derivative(axes(L,1))
    Δ = -(B'D'D*B) # Weak Laplacian

    f = L*exp.(L.points) # project exp(x)
    u = B * (Δ \ (B'f))

    @test u[0.1] ≈ -0.06612902692412974
end


@testset "Helmholtz" begin
    L = LinearSpline(range(0,stop=1,length=10))
    B = L[:,2:end-1] # Zero dirichlet by dropping first and last spline
    D = Derivative(axes(L,1))
    A = -(B'D'D*B) + 100^2*B'B # Weak Laplacian

    f = L*exp.(L.points) # project exp(x)
    u = B * (A \ (B'f))

    @test u[0.1] ≈ 0.00012678835289369413
end


S = Jacobi(true,true)
W = Diagonal(JacobiWeight(true,true))
D = Derivative(axes(W,1))


S'W'W*S

N = 10
L = D*W*S[:,1:N]
    # temporary work around to force 3-term materialize
    L = *(L.factors[1:3]...) * L.factors[4]

Δ = L'L # weak second derivative

f = Legendre() * Vcat(1:2, Zeros(∞))
L'f

A = (L').factors[2]*f.factors[1]

A*f.factors[2]

axes(L')

Legendre()'f

A = Diagonal(1:∞)


@which A*x
1
using InfiniteArrays, BandedMatrices, InteractiveUtils, Test

A = BandedMatrices._BandedMatrix((1:∞)', ∞, -1,1)
x = Vcat(1:3, Zeros(∞))

 Zeros(∞) ==  Zeros(∞)


@test A*x == Vcat([4.0,9.0], Zeros(∞))

y = similar(Mul

y = Vcat(randn(9), Zeros(∞))
copyto!(y , Mul(A,x))

MemoryLayout(x)

similar(Mul(A,x), Float64)


typeof( Mul(B,x))

Mul2{<:BandedColumnMajor,<:Any,<:AbstractMatrix,
        <:Vcat{<:AbstractVector,<:Zeros}}





typeof(V)
V = view(bandeddata(B),1:1,3:∞)
    _BandedMatrix(V, ∞, 2,-2)

using InteractiveUtils

B


B*x

N = 100000; @time B[1:N,1:N]



import LazyArrays: MemoryLayout
MemoryLayout.(x.arrays)
