using ContinuumArrays, QuasiArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, FastTransforms, InfiniteArrays, Test, Base64, RecipesBase
import ContinuumArrays: ℵ₁, materialize, AffineQuasiVector, BasisLayout, AdjointBasisLayout, SubBasisLayout, ℵ₁,
                        MappedBasisLayout, AdjointMappedBasisLayout, MappedWeightedBasisLayout, TransformFactorization, Weight, WeightedBasisLayout, SubWeightedBasisLayout, WeightLayout,
                        Expansion, basis, invmap, Map, checkpoints
import QuasiArrays: SubQuasiArray, MulQuasiMatrix, Vec, Inclusion, QuasiDiagonal, LazyQuasiArrayApplyStyle, LazyQuasiArrayStyle
import LazyArrays: MemoryLayout, ApplyStyle, Applied, colsupport, arguments, ApplyLayout, LdivStyle, MulStyle


@testset "Inclusion" begin
    @testset "basics" begin
        x = Inclusion(-1..1)
        @test eltype(x) == Float64
        @test x[0.1] ≡ 0.1
        @test x[0] ≡ x[0.0] ≡ 0.0
    end

    @testset "broadcast" begin
        x = Inclusion(-1.0..1)
        X = QuasiDiagonal(x)
        @test X[-1:0.1:1,-1:0.1:1] == Diagonal(-1:0.1:1)
        @test Base.BroadcastStyle(typeof(x)) == LazyQuasiArrayStyle{1}()
        @test x .* x isa BroadcastQuasiArray
        @test (x.*x)[0.1] == 0.1^2

        @test exp.(x)[1.0] == exp(1.0)
        @test exp.(-x)[1.0] == exp(-1.0)
        @test exp.(x.^2)[1.0] == exp(1.0)
        @test exp.(-x.^2/(20/2))[1.0] == exp(-1.0^2/(20/2))

        @test dot(x,x) ≈ 2/3
        @test norm(x) ≈ sqrt(2/3)
    end

    @testset "Derivative" begin
        x = Inclusion(-1..1)
        D = Derivative(x)
        @test D*x ≡ QuasiOnes(x)
        @test D^2 * x ≡ QuasiZeros(x)
        @test D*[x D*x] == [D*x D^2*x]
    end
end

include("test_maps.jl")

@testset "DiracDelta" begin
    δ = DiracDelta()
    @test δ == DiracDelta(0, Inclusion(0))
    @test axes(δ) ≡ (axes(δ,1),) ≡ (Inclusion(0.0),)
    @test size(δ) ≡ (length(δ),) ≡ (1,)
    @test_throws BoundsError δ[1.1]
    @test δ[0.0] ≡ Inf
    @test Base.IndexStyle(δ) ≡ Base.IndexLinear()

    δ = DiracDelta(0, -1..3)
    @test δ == DiracDelta{Float64}(0, -1..3)
    @test axes(δ) ≡ (axes(δ,1),) ≡ (Inclusion(-1..3),)
    @test size(δ) ≡ (length(δ),) ≡ (ℵ₁,)
    @test δ[1.1] ≡ 0.0
    @test δ[0.0] ≡ Inf
    @test Base.IndexStyle(δ) ≡ Base.IndexLinear()

    @test stringmime("text/plain", δ) == "δ at 0.0 over Inclusion(-1..3)"
    x = Inclusion(-1..1)
    @test stringmime("text/plain", δ[2x .+ 1]) == "δ at 0.0 over Inclusion(-1..3) affine mapped to -1..1"
end

@testset "Splines" begin
    @testset "HeavisideSpline" begin
        H = HeavisideSpline([1,2,3])

        @test axes(H) ≡ (axes(H,1),axes(H,2)) ≡ (Inclusion(1..3), Base.OneTo(2))
        @test size(H) ≡ (size(H,1),size(H,2)) ≡ (ℵ₁, 2)

        @test_throws BoundsError H[0.1, 1]
        @test H[1.1,1] ≡ H'[1,1.1] ≡ transpose(H)[1,1.1] ≡ 1.0
        @test H[2.1,1] ≡ H'[1,2.1] ≡ transpose(H)[1,2.1] ≡ 0.0
        @test H[1.1,2] ≡ 0.0
        @test H[2.1,2] ≡ 1.0
        @test_throws BoundsError H[2.1,3]
        @test_throws BoundsError H'[3,2.1]
        @test_throws BoundsError transpose(H)[3,2.1]
        @test_throws BoundsError H[3.1,2]

        @test all(H[[1.1,2.1], 1] .=== H'[1,[1.1,2.1]] .=== transpose(H)[1,[1.1,2.1]] .=== [1.0,0.0])
        @test all(H[1.1,1:2] .=== H[1.1,:] .=== [1.0,0.0])
        @test all(H[[1.1,2.1], 1:2] .=== [1.0 0.0; 0.0 1.0])

        @test_throws BoundsError H[[0.1,2.1], 1]
        @test MemoryLayout(typeof(H)) == BasisLayout()
        @test ApplyStyle(*, typeof(H), typeof([1,2])) isa MulStyle

        f = H*[1,2]
        @test f isa ApplyQuasiArray
        @test axes(f) == (Inclusion(1.0..3.0),)
        @test f[1.1] ≈ 1
        @test f[2.1] ≈ 2

        @test @inferred(H'H) == @inferred(materialize(applied(*,H',H))) == Eye(2)
        if VERSION < v"1.6-"
            @test summary(f) == "(Spline{0,Float64,Array{$Int,1}}) * (2-element Array{$Int,1})"
            @test stringmime("text/plain", f) == "Spline{0,Float64,Array{$Int,1}} * [1, 2]"
        else
            @test summary(f) == "(HeavisideSpline{Float64, Vector{$Int}}) * (2-element Vector{$Int})"
            @test stringmime("text/plain", f) == "HeavisideSpline{Float64, Vector{$Int}} * [1, 2]"
        end
    end

    @testset "LinearSpline" begin
        @testset "Evaluation" begin
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
        end

        @testset "Expansion" begin
            L = LinearSpline([1,2,3])
            f = L*[1,2,4]

            @test basis(f) == L
            @test axes(f) == (Inclusion(1.0..3.0),)
            @test f[1.1] ≈ 1.1
            @test f[2.1] ≈ 2.2

            δ = DiracDelta(1.2,1..3)
            L = LinearSpline([1,2,3])
            @test @inferred(δ'L) ≈ [0.8, 0.2, 0.0]

            @test @inferred(L'L) == SymTridiagonal([1/3,2/3,1/3], [1/6,1/6])

            @testset "Algebra" begin
                @test 2f == f*2 == 2 .* f == f .* 2
                @test 2\f == f/2 == 2 .\ f == f ./ 2
                @test sum(f) ≈ 4.5
            end
        end

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
            @test MemoryLayout(L') == AdjointBasisLayout()
            @test @inferred([3,4,5]'*L') isa ApplyQuasiArray
        end

        @testset "+" begin
            L = LinearSpline([1,2,3])
            b = L*[3,4,5] + L*[1.,2,3]
            @test ApplyStyle(\, typeof(L), typeof(b)) == LdivStyle()
            @test (L\b) == [4,6,8]
            B = BroadcastQuasiArray(+, L, L)
            @test L\B == 2Eye(3)

            b = L*[3,4,5] - L*[1.,2,3]
            @test (L\b) == [2,2,2]
            B = BroadcastQuasiArray(-, L, L)
            @test L\B == 0Eye(3)
        end
    end

    @testset "Algebra" begin
        L = LinearSpline([1,2,3])
        f = L*[1,2,4]
        g = L*[5,6,7]

        @test f isa Expansion
        @test 2f isa Expansion
        @test f*2 isa Expansion
        @test 2\f isa Expansion
        @test f/2 isa Expansion
        @test f+g isa Expansion
        @test f-g isa Expansion
        @test f[1.2] == 1.2
        @test (2f)[1.2] == (f*2)[1.2] == 2.4
        @test (2\f)[1.2] == (f/2)[1.2] == 0.6
        @test (f+g)[1.2] ≈ f[1.2] + g[1.2]
        @test (f-g)[1.2] ≈ f[1.2] - g[1.2]
    end

    @testset "Derivative" begin
        L = LinearSpline([1,2,3])
        f = L*[1,2,4]

        D = Derivative(axes(L,1))
        @test copy(D) == D

        @test D*L isa MulQuasiMatrix
        @test length((D*L).args) == 2
        @test eltype(D*L) == Float64

        M = applied(*, (D*L).args..., [1,2,4])
        @test eltype(materialize(M)) == Float64

        M = applied(*, D, L, [1,2,4])
        @test materialize(M) isa ApplyQuasiArray

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


        @testset "View derivatives" begin
            L = LinearSpline(1:5)
            H = HeavisideSpline(1:5)
            x = axes(L,1)
            D = Derivative(x)

            @test view(D, x, x) ≡ D
            @test_throws BoundsError view(D, Inclusion(0..1), x)
            jr = [1,3,4]
            @test H \ (D*L[:,jr]) == H\(L[:,jr]'D')' == (H \ (D*L))[:,jr]

            @test D*L[:,jr]*[1,2,3] == D * L * [1,0,2,3,0]
            @test L[:,jr]'D'D*L[:,jr] == (L'D'D*L)[jr,jr]

            a = affine(0..1, 1..5)
            D̃ = Derivative(axes(a,1))
            @test H[a,:] \ (D̃ * L[a,:]) == H[a,:] \ (L[a,:]'D̃')' ==  4*(H\(D*L))
            @test_throws DimensionMismatch D * L[a,:]
            @test_throws DimensionMismatch D̃ * L[:,jr]

            @test ContinuumArrays.simplifiable(*, D, L[:,jr]) isa Val{true}
            @test ContinuumArrays.simplifiable(*, L[:,jr]', D') isa Val{true}
            @test ContinuumArrays.simplifiable(*, D̃, L[a,jr]) isa Val{true}
            @test ContinuumArrays.simplifiable(*, L[a,jr]', D̃') isa Val{true}
        end
    end

    @testset "Weak Laplacian" begin
        H = HeavisideSpline(0:2)
        L = LinearSpline(0:2)
        D = Derivative(axes(L,1))

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
        @test L[:,2:3]\L isa BandedMatrix
        @test L[:,2:3]\L == [0 1 0 0; 0 0 1 0]

        @testset "Subindex of splines" begin
            L = LinearSpline(range(0,stop=1,length=10))
            @test L[:,2:end-1] isa SubQuasiArray
            @test L[:,2:end-1][0.1,1] == L[0.1,2]
            v = randn(8)
            f = L[:,2:end-1] * v
            @test f[0.1] ≈ (L*[0; v; 0])[0.1]
        end

        @testset "sub-of-sub" begin
            L = LinearSpline([1,2,3])
            V = view(L,:,1:2)
            @test V == V
            V2 = view(V,1.1:0.1:2,:)
            @test V2 == L[1.1:0.1:2,1:2]
        end

        @testset "sub-colon" begin
            L = LinearSpline([1,2,3])
            @test L[:,1][1.1] == (L')[1,:][1.1] == L[1.1,1]
            @test L[:,1:2][1.1,:] == (L')[1:2,:][:,1.1] == L[1.1,1:2]
        end

        @testset "transform" begin
            L = LinearSpline([1,2,3])
            x = axes(L,1)
            @test (L \ x) == [1,2,3]
            @test factorize(L[:,2:end-1]) isa ContinuumArrays.ProjectionFactorization
            @test L[:,1:2] \ x == [1,2]
            @test L \ [x one(x)] ≈ [L\x L\one(x)]
            @test factorize(L) \ QuasiOnes(x, Base.OneTo(3)) ≈ L \ QuasiOnes(x, Base.OneTo(3)) ≈ ones(3,3)

            L = LinearSpline(range(0,1; length=10_000))
            x = axes(L,1)
            @test L[0.123,:]'* (L \ exp.(x)) ≈ exp(0.123) atol=1E-9
            @test L[0.123,2:end-1]'* (L[:,2:end-1] \ exp.(x)) ≈ exp(0.123) atol=1E-9
        end
    end

    @testset "sum" begin
        H = HeavisideSpline([1,2,3,6])
        @test sum(H; dims=1) * [1,1,1] == [5]
        x = Inclusion(0..1)
        B = H[5x .+ 1,:]
        @test sum(B; dims=1) * [1,1,1] == [1]
        @test sum(H[:,1:2]; dims=1) * [1,1] == [2]
        @test sum(H'; dims=2) == permutedims(sum(H; dims=1))

        u = H * randn(3)
        @test sum(u[5x .+ 1]) ≈ sum(view(u,5x .+ 1)) ≈ sum(u)/5

        L = LinearSpline([1,2,3,6])
        D = Derivative(axes(L,1))
        @test sum(D*L; dims=1) ≈ sum((D*L)'; dims=2)' ≈ [-1 zeros(1,2) 1]
    end

    @testset "Poisson" begin
        L = LinearSpline(range(0,stop=1,length=10))
        B = L[:,2:end-1] # Zero dirichlet by dropping first and last spline
        @test B'B == (L'L)[2:end-1,2:end-1]
        D = Derivative(axes(L,1))
        @test apply(*,D,B) isa MulQuasiMatrix
        @test D*B isa MulQuasiMatrix
        @test apply(*,D,B)[0.1,1] == (D*B)[0.1,1] == 9
        @test apply(*,D,B)[0.2,1] == (D*B)[0.2,1] == -9
        Δ = -((D*B)'*(D*B)) # Weak Laplacian
        @test Δ isa BandedMatrix

        @test B'D' isa MulQuasiMatrix
        @test length((B'D').args) == 2

        @test *(B',D',D,B) isa BandedMatrix

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

    @testset "Change-of-variables" begin
        x = Inclusion(0..1)
        y = 2x .- 1
        L = LinearSpline(range(-1,stop=1,length=10))
        @test L[y,:][0.1,:] == L[2*0.1-1,:]
        @test L[y,:]'L[y,:] isa SymTridiagonal
        @test L[y,:]'L[y,:] == 1/2*(L'L)

        D = Derivative(axes(L,1))
        H = HeavisideSpline(L.points)
        @test H\((D*L) * 2) ≈ (H\(D*L))*2 ≈ diagm(0 => fill(-9,9), 1 => fill(9,9))[1:end-1,:]

        @test MemoryLayout(L[y,:]) isa MappedBasisLayout
        @test MemoryLayout(L[y,:]') isa AdjointMappedBasisLayout
        a,b = arguments((D*L)[y,:])
        @test H[y,:]\a == Eye(9)
        @test H[y,:] \ (D*L)[y,:] isa BandedMatrix
        @test @inferred(grid(L[y,:])) ≈ (grid(L) .+ 1) ./ 2

        D = Derivative(x)
        @test (D*L[y,:])[0.1,1] ≈ -9
        @test H[y,:] \ (D*L[y,:]) isa BandedMatrix
        @test H[y,:] \ (D*L[y,:]) ≈ diagm(0 => fill(-9,9), 1 => fill(9,9))[1:end-1,:]

        B = L[y,2:end-1]
        @test MemoryLayout(typeof(B)) isa MappedBasisLayout
        @test B[0.1,1] == L[2*0.1-1,2]
        @test B\B == Eye(8)
        @test L[y,:] \ B == Eye(10)[:,2:end-1]
        @test B'B == (1/2)*(L'L)[2:end-1,2:end-1]

        @testset "algebra" begin
            f = L[y,:] * randn(10)
            g = L[y,:] * randn(10)
            @test f + g isa Expansion
            @test (f+g)[0.1] ≈ f[0.1] + g[0.1]
        end

        @testset "vec demap" begin
            @test L[y,:] \ exp.(axes(L,1))[y] ≈ L[y,:] \ exp.(y) ≈  factorize(L[y,:]) \ exp.(y)
            @test ContinuumArrays.demap(view(axes(L,1),y)) == axes(L,1)
        end
    end

    @testset "diff" begin
        L = LinearSpline(range(-1,stop=1,length=10))
        f = L * randn(size(L,2))
        h = 0.0001;
        @test diff(f)[0.1] ≈ (f[0.1+h]-f[0.1])/h
    end

    @testset "show" begin
        x = Inclusion(0..1)
        H = HeavisideSpline([1,2,3,6])
        B = H[5x .+ 1,:]
        u = H * [1,2,3]
        if VERSION < v"1.6-"
            @test stringmime("text/plain", B) == "Spline{0,Float64,Array{$Int,1}} affine mapped to 0..1"
        else
            @test stringmime("text/plain", B) == "HeavisideSpline{Float64, Vector{$Int}} affine mapped to 0..1"
        end
    end
end

@testset "Kernels" begin
    x = Inclusion(0..1)
    K = x .- x'
    @test K[0.1,0.2] == K[Inclusion(0..0.5), Inclusion(0..0.5)][0.1,0.2] == 0.1 - 0.2
    @test_throws BoundsError K[Inclusion(0..0.5), Inclusion(0..0.5)][1,1]
end

"""
This is a simple implementation of Chebyshev for testing. Use OrthogonalPolynomialsQuasi
for the real implementation.
"""
struct Chebyshev <: Basis{Float64}
    n::Int
end

struct ChebyshevWeight <: Weight{Float64} end

Base.:(==)(::Chebyshev, ::Chebyshev) = true
Base.axes(T::Chebyshev) = (Inclusion(-1..1), Base.OneTo(T.n))
ContinuumArrays.grid(T::Chebyshev) = chebyshevpoints(Float64, T.n, Val(1))
Base.axes(T::ChebyshevWeight) = (Inclusion(-1..1),)

Base.getindex(::Chebyshev, x::Float64, n::Int) = cos((n-1)*acos(x))
Base.getindex(::ChebyshevWeight, x::Float64) = 1/sqrt(1-x^2)
Base.getindex(w::ChebyshevWeight, ::Inclusion) = w # TODO: make automatic

LinearAlgebra.factorize(L::Chebyshev) =
    TransformFactorization(grid(L), plan_chebyshevtransform(Array{Float64}(undef, size(L,2))))

# This is wrong but just for tests
Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::Expansion{<:Any,<:Chebyshev}, b::Chebyshev) = b * Matrix(I, 5, 5)

struct QuadraticMap{T} <: Map{T} end
struct InvQuadraticMap{T} <: Map{T} end

QuadraticMap() = QuadraticMap{Float64}()
InvQuadraticMap() = InvQuadraticMap{Float64}()

Base.getindex(::QuadraticMap, r::Number) = 2r^2-1
Base.axes(::QuadraticMap{T}) where T = (Inclusion(0..1),)
Base.axes(::InvQuadraticMap{T}) where T = (Inclusion(-1..1),)
Base.getindex(d::InvQuadraticMap, x::Number) = sqrt((x+1)/2)
ContinuumArrays.invmap(::QuadraticMap{T}) where T = InvQuadraticMap{T}()
ContinuumArrays.invmap(::InvQuadraticMap{T}) where T = QuadraticMap{T}()


@testset "Chebyshev" begin
    T = Chebyshev(5)
    w = ChebyshevWeight()
    wT = w .* T
    x = axes(T,1)

    @testset "basics" begin
        F = factorize(T)
        g = grid(F)
        @test T \ exp.(x) == F \ exp.(x) == F \ exp.(g) == chebyshevtransform(exp.(g), Val(1))
        @test all(checkpoints(T) .∈ Ref(axes(T,1)))

        @test F \ [exp.(x) cos.(x)] ≈ [F \ exp.(x) F \ cos.(x)]
    end

    @testset "Weighted" begin
        @test MemoryLayout(w) isa WeightLayout
        @test MemoryLayout(w[Inclusion(0..1)]) isa WeightLayout

        wT2 = w .* T[:,2:4]
        wT3 = wT[:,2:4]
        @test MemoryLayout(wT) == WeightedBasisLayout()
        @test MemoryLayout(wT2) == WeightedBasisLayout()
        @test MemoryLayout(wT3) == SubWeightedBasisLayout()
        @test grid(wT) == grid(wT2) == grid(wT3) == grid(T)

        @test ContinuumArrays.unweightedbasis(wT) ≡ T
        @test ContinuumArrays.unweightedbasis(wT2) ≡ T[:,2:4]
        @test ContinuumArrays.unweightedbasis(wT3) ≡ T[:,2:4]

        @test ContinuumArrays.weight(wT) ≡ ContinuumArrays.weight(wT2) ≡ ContinuumArrays.weight(wT3) ≡ w

        @test wT \ @.(exp(x) / sqrt(1-x^2)) ≈ T \ exp.(x)
    end
    @testset "Mapped" begin
        y = affine(0..1, x)

        @test summary(T[y,:]) == "Chebyshev affine mapped to 0..1"
        @test MemoryLayout(wT[y,:]) isa MappedWeightedBasisLayout
        @test MemoryLayout(w[y] .* T[y,:]) isa MappedWeightedBasisLayout
        @test wT[y,:][[0.1,0.2],1:5] == (w[y] .* T[y,:])[[0.1,0.2],1:5] == (w .* T[:,1:5])[y,:][[0.1,0.2],:]
        @test MemoryLayout(wT[y,1:3]) isa MappedWeightedBasisLayout
        @test wT[y,1:3][[0.1,0.2],1:2] == wT[y[[0.1,0.2]],1:2]

        @testset "QuadraticMap" begin
            m = QuadraticMap()
            mi = InvQuadraticMap()
            @test 0.1 ∈ m
            @test -0.1 ∈ m
            @test 2 ∉ m
            @test 0.1 ∈ mi
            @test -0.1 ∉ mi

            @test m[findfirst(isequal(0.1), m)] ≈ 0.1
            @test m[findlast(isequal(0.1), m)] ≈ 0.1
            @test m[findall(isequal(0.1), m)] ≈ [0.1]

            @test m[Inclusion(0..1)] ≡ m
            @test_throws BoundsError m[Inclusion(-1..1)]
            T = Chebyshev(5)
            M = T[m,:]
            @test MemoryLayout(M) isa MappedBasisLayout
            @test MemoryLayout(M[:,1:3]) isa MappedBasisLayout
            @test M[0.1,:] ≈ T[2*0.1^2-1,:]
            x = axes(M,1)
            @test x == Inclusion(0..1)
            @test M \ exp.(x) ≈ T \ exp.(sqrt.((axes(T,1) .+ 1)/2))
        end
    end

    @testset "Broadcasted" begin
        T = Chebyshev(5)
        F = factorize(T)
        x = axes(T,1)
        a = 1 .+ x .+ x.^2
        # The following are wrong, just testing dispatch
        @test T \ (a .* T) == I
        @test T \ (a .* (T * (T \ a))) ≈ [2.875, 3.5, 2.0, 0.5, 0.125]
        f = exp.(x) .* a # another broadcast layout
        @test T \ f == F \ f

        ã = T * (T \ a)
        @test T \ (ã .* ã) ≈ [1.5,1,0.5,0,0]
    end
end

include("test_basisconcat.jl")

@testset "Plotting" begin
    L = LinearSpline(0:5)
    rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), L)
    @test rep[1].args == (L.points,L[L.points,:])

    u = L*randn(6)
    rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
    @test rep[1].args == (L.points,u[L.points])
end