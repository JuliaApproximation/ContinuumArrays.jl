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

A = @inferred(PInv(Jacobi(2,2))*D*S)
@test A isa BandedMatrix
@test size(A) == (∞,∞)
@test A[1:10,1:10] == diagm(1 => 1:0.5:5)

M = @inferred(D*S)
@test M isa Mul
@test M.factors[1] == Jacobi(2,2)
@test M.factors[2][1:10,1:10] == A[1:10,1:10]


L = Diagonal(JacobiWeight(true,false))
A = @inferred(PInv(Jacobi(false,true))*L*S)
@test A isa BandedMatrix
@test size(A) == (∞,∞)

L = Diagonal(JacobiWeight(false,true))
A = @inferred(PInv(Jacobi(true,false))*L*S)
@test A isa BandedMatrix
@test size(A) == (∞,∞)


L = Diagonal(JacobiWeight(true,true))


L = Legendre()

A, B = (L'L),(PInv(L)*W*S)

M = A*B
    @which M.mul[1,1]

@which materialize(Mul(A,B))

M = LazyArrays.MulArray(Mul(A,B))
    axes(M)

@time A[1:100000,1:100000]
@profiler A.data[:,1:10_000]

V = view(A.data,:,1:100)

N = 10_000; M = randn(3,N)

A.data.arrays[2]

@time begin
    M[1,:] .= view(A.data.arrays[1]', 1:N)
    M[2,:] .= view(A.data.arrays[2], 1, 1:N)
    M[3,:] .= view(A.data.arrays[3]', 1:N)
end
M

@which copyto!(M, V)

@time

typeof(V)

using Profile
Profile.clear()
@time randn(3,10_000)

W*S




M = Mul(S',W',W,S)
materialize(M)
materialize(Mul(S',W',W))

@which materialize(M)

W*W

S'W'W*S

N = 10
L = D*W*S[:,1:N]
# temporary work around to force 3-term materialize
    L = *(L.factors[1:3]...) * L.factors[4]


*(L.factors[1:3]...)

@test L.factors isa Tuple{<:Legendre,<:BandedMatrix,<:BandedMatrix}

Δ = L'L # weak second derivative
@test size(Δ) == (10,10)


Vcat(1:2, Zeros(∞))
import Base.Broadcast: broadcasted
@which broadcasted(*, Fill(2,∞) , Vcat(1:2, Zeros(∞)))
(1:∞) .* Vcat(1:2, Zeros(∞))

(1:10) .* Zeros(10)
A,B = (1:∞) , Vcat(1:2, Zeros(∞))

kr = LazyArrays._vcat_axes(axes.(B.arrays)...)
A_arrays = LazyArrays._vcat_getindex_eval(A,kr...)

broadcast(*, A_arrays[1], B.arrays[1])

A.*B

_Vcat(broadcast((a,b) -> broadcast(op,a,b), A_arrays, B.arrays))

f = Legendre() * Vcat(randn(20), Zeros(∞))
@time Vector(L'f)

A,v=(L'f).factors[end-1:end]

import LazyArrays: MemoryLayout
@which MemoryLayout(v)


v
A*v

LazyArrays.MemoryLayout(A)

A = (L').factors[2]*f.factors[1]

A*f.factors[2]

axes(L')

Legendre()'f

A = Diagonal(1:∞)


@which A*x
1
using InfiniteArrays, BandedMatrices, InteractiveUtils, Test

A = BandedMatrices._BandedMatrix((1:∞)', ∞, -1,1)
D = Diagonal(1:∞)
x = Vcat(randn(100000), Zeros(∞))
@time A*x
@time D*x

M = Mul(A,x)

materialize(M)

similar(M)



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
