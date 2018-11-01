using ContinuumArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, Test
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
    B1 = L[:,1]
    @test B1 isa SubQuasiArray{Float64,1}
    @test size(B1) == (ℵ₁,)
    @test B1[0.1] == L[0.1,1]
    @test_throws BoundsError B1[2.2]

    B = L[:,1:2]
    @test B isa SubQuasiArray{Float64,2}
    @test B[0.1,:] == L[0.1,1:2]

    B = L[:,2:end-1]
    @test B[0.1,:] == [0.1]
end

A = randn(4,4)
@which lastindex(A,2)

@time begin
L = LinearSpline(range(0,stop=1,length=10))[:,2:end-1]
D = Derivative(axes(L,1))

D*L

Derivative(0..1)*parent(L)




M = Mul(D,L)
A, B = M.factors
axes(A,2) == axes(B,1) || throw(DimensionMismatch())
P = parent(B)
(Derivative(axes(P,1))*P)[parentindices(P)...]

@which axes(D,2)

axes(D,2)
D*L
A = -(L'D'D*L)
f = L*exp.(L.points)

A \ (L'f)

u = A[2:end-1,2:end-1] \ (L'f)[2:end-1]

cond(A[2:end-1,2:end-1])

A[2:end-1,2:end-1] *u - (L'f)[2:end-1]

v = L*[0; u; 0]


v[0.2]
using Plots
plot(0:0.01:1,getindex.(Ref(v),0:0.01:1))
    plot!(u1)
ui

using ApproxFun

x = Fun(0..1)
    u1 = [Dirichlet(Chebyshev(0..1)); ApproxFun.Derivative()^2] \ [[0,0], exp(x)]

plot(u1)

u_ex = L*u1.(L.points)

xx = 0:0.01:1;
    plot(xx,getindex.(Ref(D*u_ex),xx))

getindex.(Ref(D*u_ex),xx)
plot!(u1')

f[0.1]-exp(0.1)

(D*v)'*(D*v)



L'f





(L'f)

(L'f)

end
