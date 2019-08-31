using ContinuumArrays, QuasiArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, Test, InfiniteArrays
import ContinuumArrays: ℵ₁, materialize, Chebyshev, Ultraspherical, jacobioperator, SimplifyStyle
import QuasiArrays: SubQuasiArray, MulQuasiMatrix, Vec, Inclusion, QuasiDiagonal, LazyQuasiArrayApplyStyle, LmaterializeApplyStyle
import LazyArrays: MemoryLayout, ApplyStyle, Applied, colsupport


@testset "Inclusion" begin
    @test_throws InexactError Inclusion(-1..1)[0.1]
    @test Inclusion(-1..1)[0.0] === 0
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

@testset "Ultraspherical" begin
    T = Chebyshev()
    U = Ultraspherical(1)
    C = Ultraspherical(2)
    D = Derivative(axes(T,1))

    @test T\T === pinv(T)*T === Eye(∞)
    @test U\U === pinv(U)*U === Eye(∞)
    @test C\C === pinv(C)*C === Eye(∞)

    @test ApplyStyle(\,typeof(U),typeof(applied(*,D,T))) == SimplifyStyle()
    @test materialize(@~ U\(D*T)) isa BandedMatrix
    D₀ = U\(D*T)
    @test_broken D₀ isa BandedMatrix
    @test D₀[1:10,1:10] isa BandedMatrix{Float64}
    @test D₀[1:10,1:10] == diagm(1 => 1:9)
    @test colsupport(D₀,1) == 1:0

    D₁ = C\(D*U)
    @test D₁ isa BandedMatrix
    @test apply(*,D₁,D₀.args...)[1:10,1:10] == diagm(2 => 4:2:18)
    @test (D₁*D₀)[1:10,1:10] == diagm(2 => 4:2:18)

    S₀ = (U\T)[1:10,1:10]
    @test S₀ isa BandedMatrix{Float64}
    @test S₀ == diagm(0 => [1.0; fill(0.5,9)], 2=> fill(-0.5,8))

    S₁ = (C\U)[1:10,1:10]
    @test S₁ isa BandedMatrix{Float64}
    @test S₁ == diagm(0 => 1 ./ (1:10), 2=> -(1 ./ (3:10)))
end

@testset "Jacobi" begin
    S = Jacobi(true,true)
    W = Diagonal(JacobiWeight(true,true))
    D = Derivative(axes(W,1))
    P = Legendre()

    @test pinv(pinv(S)) === S
    @test P\P === pinv(P)*P === Eye(∞)

    Bi = pinv(Jacobi(2,2))
    @test Bi isa QuasiArrays.PInvQuasiMatrix

    A = apply(\, Jacobi(2,2), applied(*, D, S))
    @test A isa BandedMatrix
    A = Jacobi(2,2) \ (D*S)
    @test typeof(A) == typeof(pinv(Jacobi(2,2))*(D*S))

    @test A isa MulMatrix
    @test isbanded(A)
    @test bandwidths(A) == (-1,1)
    @test size(A) == (∞,∞)
    @test A[1:10,1:10] == diagm(1 => 1:0.5:5)

    @test_broken @inferred(D*S)
    M = D*S
    @test M isa MulQuasiMatrix
    @test M.args[1] == Jacobi(2,2)
    @test M.args[2][1:10,1:10] == A[1:10,1:10]

    L = Diagonal(JacobiWeight(true,false))
    @test apply(\, Jacobi(false,true), applied(*,L,S)) isa BandedMatrix
    @test_broken @inferred(Jacobi(false,true)\(L*S))
    A = Jacobi(false,true)\(L*S)
    @test A isa BandedMatrix
    @test size(A) == (∞,∞)

    L = Diagonal(JacobiWeight(false,true))
    @test_broken @inferred(Jacobi(true,false)\(L*S))
    A = Jacobi(true,false)\(L*S)
    @test A isa BandedMatrix
    @test size(A) == (∞,∞)

    A,B = (P'P),P\(W*S)

    M = Mul(A,B)
    @test M[1,1] == 4/3

    M = ApplyMatrix{Float64}(*,A,B)
    M̃ = M[1:10,1:10]
    @test_broken M̃ isa BandedMatrix
    @test_broken bandwidths(M̃) == (2,0)

    @test A*B isa MulArray

    A,B,C = (P\(W*S))',(P'P),P\(W*S)
    M = ApplyArray(*,A,B,C)
    @test bandwidths(M) == (2,2)
    @test M[1,1] ≈  1+1/15
    @test typeof(M) == typeof(A*B*C)
    M = A*B*C
    @test bandwidths(M) == (2,2)
    @test M[1,1] ≈  1+1/15
end

@testset "P-FEM" begin
    S = Jacobi(true,true)
    W = Diagonal(JacobiWeight(true,true))
    D = Derivative(axes(W,1))
    P = Legendre()
    
    M = P\(D*W*S)
    @test M isa ApplyArray
    @test M[1:10,1:10] == diagm(-1 => -2.0:-2:-18.0)

    N = 10
    A = D*W*S[:,1:N]
    @test A.args[1] == P    
    @test P\((D*W)*S[:,1:N]) isa AbstractMatrix
    @test P\(D*W*S[:,1:N]) isa AbstractMatrix

    L = D*W*S
    Δ = L'L
    @test Δ isa MulMatrix
    @test_broken Δ[1:3,1:3] isa BandedMatrix
    @test bandwidths(Δ) == (0,0)

    L = D*W*S[:,1:N]

    A  = apply(*, (L').args..., L.args...)
    @test A isa MulQuasiMatrix

    A  = *((L').args..., L.args...)
    @test A isa MulQuasiMatrix

    @test apply(*,L',L) isa QuasiArrays.ApplyQuasiArray

    Δ = L'L
    @test Δ isa MulMatrix
    @test bandwidths(Δ) == (0,0)
end

@testset "Chebyshev evaluation" begin
    P = Chebyshev()
    @test @inferred(P[0.1,Base.OneTo(0)]) == Float64[]
    @test @inferred(P[0.1,Base.OneTo(1)]) == [1.0]
    @test @inferred(P[0.1,Base.OneTo(2)]) == [1.0,0.1]
    for N = 1:10
        @test @inferred(P[0.1,Base.OneTo(N)]) ≈ @inferred(P[0.1,1:N]) ≈ [cos(n*acos(0.1)) for n = 0:N-1]
        @test @inferred(P[0.1,N]) ≈ cos((N-1)*acos(0.1))
    end
    @test P[0.1,[2,5,10]] ≈ [0.1,cos(4acos(0.1)),cos(9acos(0.1))]

    P = Ultraspherical(1)
    @test @inferred(P[0.1,Base.OneTo(0)]) == Float64[]
    @test @inferred(P[0.1,Base.OneTo(1)]) == [1.0]
    @test @inferred(P[0.1,Base.OneTo(2)]) == [1.0,0.2]
    for N = 1:10
        @test @inferred(P[0.1,Base.OneTo(N)]) ≈ @inferred(P[0.1,1:N]) ≈ [sin((n+1)*acos(0.1))/sin(acos(0.1)) for n = 0:N-1]
        @test @inferred(P[0.1,N]) ≈ sin(N*acos(0.1))/sin(acos(0.1))
    end
    @test P[0.1,[2,5,10]] ≈ [0.2,sin(5acos(0.1))/sin(acos(0.1)),sin(10acos(0.1))/sin(acos(0.1))]

    P = Ultraspherical(2)
    @test @inferred(P[0.1,Base.OneTo(0)]) == Float64[]
    @test @inferred(P[0.1,Base.OneTo(1)]) == [1.0]
    @test @inferred(P[0.1,Base.OneTo(2)]) == [1.0,0.4]
    @test @inferred(P[0.1,Base.OneTo(3)]) == [1.0,0.4,-1.88]
end

@testset "Collocation" begin
    P = Chebyshev()
    D = Derivative(axes(P,1))
    n = 300
    x = cos.((0:n-2) .* π ./ (n-2))
    cfs = [P[-1,1:n]'; (D*P)[x,1:n] - P[x,1:n]] \ [exp(-1); zeros(n-1)]
    u = P[:,1:n]*cfs
    @test u[0.1] ≈ exp(0.1)

    P = Chebyshev()
    D = Derivative(axes(P,1))
    D2 = D*(D*P) # could be D^2*P in the future
    n = 300
    x = cos.((1:n-2) .* π ./ (n-1)) # interior Chebyshev points 
    C = [P[-1,1:n]';
         D2[x,1:n] + P[x,1:n];
         P[1,1:n]']
    cfs = C \ [1; zeros(n-2); 2] # Chebyshev coefficients
    u = P[:,1:n]*cfs  # interpret in basis
    @test u[0.1] ≈ (3cos(0.1)sec(1) + csc(1)sin(0.1))/2
end