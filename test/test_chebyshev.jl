using ContinuumArrays, LinearAlgebra, QuasiArrays, ArrayLayouts, Base64, LazyArrays, Test
using FastTransforms
import ContinuumArrays: Basis, Weight, Map, LazyQuasiArrayStyle, TransformFactorization,
                        ExpansionLayout, checkpoints, MappedBasisLayout, MappedWeightedBasisLayout,
                        SubWeightedBasisLayout, WeightedBasisLayout, WeightLayout, basis, grammatrix

using IntervalSets: AbstractInterval
"""
This is a simple implementation of Chebyshev for testing. Use ClassicalOrthogonalPolynomials
    for the real implementation.
"""
struct Chebyshev <: Basis{Float64}
    n::Int
end

struct ChebyshevWeight <: Weight{Float64} end

Base.:(==)(::Chebyshev, ::Chebyshev) = true
Base.:(==)(::ChebyshevWeight, ::ChebyshevWeight) = true
Base.axes(T::Chebyshev) = (Inclusion(-1..1), Base.OneTo(T.n))
ContinuumArrays.grid(T::Chebyshev, ::Integer) = chebyshevpoints(Float64, T.n, Val(1))
Base.axes(T::ChebyshevWeight) = (Inclusion(-1..1),)

Base.getindex(::Chebyshev, x::Float64, n::Int) = cos((n-1)*acos(x))
Base.getindex(::ChebyshevWeight, x::Float64) = 1/sqrt(1-x^2)
Base.getindex(w::ChebyshevWeight, ::Inclusion) = w # TODO: make automatic

ContinuumArrays.plan_transform(L::Chebyshev, szs::NTuple{N,Int}, dims=1:N) where N = plan_chebyshevtransform(Array{eltype(L)}(undef, szs...), dims)
ContinuumArrays.basis_axes(::Inclusion{<:Any,<:AbstractInterval}, v) = Chebyshev(100)
function ContinuumArrays._sum(T::Chebyshev, dims)
    n = 2:size(T,2)-1
    [2; 0; @. ((1/(n+1) - 1/(n-1)) - ((-1)^(n+1)/(n+1) - (-1)^(n-1)/(n-1)))/2]'
end

Base.diff(T::Chebyshev; dims=1) = T # not correct but just checks expansion works

# This is wrong but just for tests
QuasiArrays.layout_broadcasted(::Tuple{ExpansionLayout,Any}, ::typeof(*), a::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{Chebyshev,Any}}, b::Chebyshev) = b * Matrix(I, 5, 5)

function ContinuumArrays.grammatrix(A::Chebyshev)
    m = size(A,2)
    T = eltype(A)
    f = (k,j) -> isodd(j-k) ? zero(T) : -(((1 + (-1)^(j + k))*(-1 + j^2 + k^2))/(j^4 + (-1 + k^2)^2 - 2j^2*(1 + k^2)))
    f.(0:m-1, (0:m-1)')
end


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

struct InfMap{T} <: Map{T}
    s
end
struct InvInfMap{T} <: Map{T}
    s
end

InfMap(s=1) = InfMap{Float64}(s)
InvInfMap(s=1) = InvInfMap{Float64}(s)

Base.getindex(m::InfMap, r::Number) = 1-2/(m.s * r+1)
Base.axes(m::InfMap{T}) where T = (Inclusion(m.s * (0..Inf)),)
Base.axes(::InvInfMap{T}) where T = (Inclusion(-1..1),)
Base.getindex(m::InvInfMap, x::Number) = m.s*( 2/(1-x) - 1)
ContinuumArrays.invmap(m::InfMap{T}) where T = InvInfMap{T}(m.s)
ContinuumArrays.invmap(m::InvInfMap{T}) where T = InfMap{T}(m.s)

struct BiInfMap{T} <: Map{T} end
struct InvBiInfMap{T} <: Map{T} end

BiInfMap() = BiInfMap{Float64}()
InvBiInfMap() = InvBiInfMap{Float64}()

Base.getindex(m::BiInfMap, y::Number) = iszero(y) ? y : (-1 + sqrt(1 + 4y^2))/(2y)
Base.axes(m::BiInfMap{T}) where T = (Inclusion(-Inf..Inf),)
Base.axes(::InvBiInfMap{T}) where T = (Inclusion(-1..1),)
Base.getindex(m::InvBiInfMap, x::Number) = x/(1-x^2)
ContinuumArrays.invmap(m::BiInfMap{T}) where T = InvBiInfMap{T}()
ContinuumArrays.invmap(m::InvBiInfMap{T}) where T = BiInfMap{T}()


struct FooDomain end

struct FooBasis  <: Basis{Float64} end
Base.axes(::FooBasis) = (Inclusion(-1..1), Base.OneTo(5))
Base.:(==)(::FooBasis, ::FooBasis) = true


@testset "Chebyshev" begin
    T = Chebyshev(5)
    w = ChebyshevWeight()
    wT = w .* T
    x = axes(T,1)

    @testset "basics" begin
        F = factorize(T)
        g = grid(F)
        @test T \ exp.(x) == F \ exp.(x) == chebyshevtransform(exp.(g), Val(1))
        @test all(checkpoints(T) .∈ Ref(axes(T,1)))

        @test T \ [exp.(x) cos.(x)] ≈ [F \ exp.(x) F \ cos.(x)]
    end

    @testset "Weighted" begin
        @test MemoryLayout(w) isa WeightLayout
        @test MemoryLayout(w[Inclusion(0..1)]) isa WeightLayout

        @test wT == wT

        wT2 = w .* T[:,2:4]
        wT3 = wT[:,2:4]
        @test MemoryLayout(wT) isa WeightedBasisLayout
        @test MemoryLayout(wT2) isa WeightedBasisLayout
        @test MemoryLayout(wT3) isa SubWeightedBasisLayout
        @test grid(wT) == grid(wT2) == grid(wT3) == grid(T)

        @test ContinuumArrays.unweighted(wT) ≡ T
        @test ContinuumArrays.unweighted(wT2) ≡ T[:,2:4]
        @test ContinuumArrays.unweighted(wT3) ≡ T[:,2:4]

        @test ContinuumArrays.weight(wT) ≡ ContinuumArrays.weight(wT2) ≡ ContinuumArrays.weight(wT3) ≡ w

        @test wT \ @.(exp(x) / sqrt(1-x^2)) ≈ T \ exp.(x)

        @test wT \ w ≈ [1; zeros(4)]

        @test (x .* wT)[0.1,:] ≈ 0.1 * wT[0.1,:]

        a = wT / wT \ @.(exp(x) / sqrt(1-x^2))
        @test wT \ (a .* T) == I # fake multiplication matrix
    end
    @testset "Mapped" begin
        y = affine(0..1, x)

        @test summary(T[y,:]) == "Chebyshev affine mapped to $(0..1)"
        @test stringmime("text/plain", T[y,:]) == "Chebyshev(5) affine mapped to $(0..1)"
        @test MemoryLayout(wT[y,:]) isa MappedWeightedBasisLayout
        @test MemoryLayout(w[y] .* T[y,:]) isa MappedWeightedBasisLayout
        @test wT[y,:][[0.1,0.2],1:5] == (w[y] .* T[y,:])[[0.1,0.2],1:5] == (w .* T[:,1:5])[y,:][[0.1,0.2],:]
        @test MemoryLayout(wT[y,1:3]) isa MappedWeightedBasisLayout
        @test MemoryLayout(wT[y,1:3][:,1:2]) isa MappedWeightedBasisLayout
        @test wT[y,1:3][[0.1,0.2],1:2] == wT[y[[0.1,0.2]],1:2]

        @test T[y,:]'T[y,:] ≈ grammatrix(T[y,:]) ≈ (T'T)/2

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

        @testset "InvMap" begin
            m = InfMap()
            mi = InvInfMap()
            @test 0.1 ∈ m
            @test -0.1 ∈ m
            @test 2 ∉ m
            @test 2 ∈ mi
            @test 0.1 ∈ mi
            @test -0.1 ∉ mi

            @test m[findfirst(isequal(0.1), m)] ≈ 0.1
            @test m[findlast(isequal(0.1), m)] ≈ 0.1
            @test m[findall(isequal(0.1), m)] ≈ [0.1]

            @test m[Inclusion(0..Inf)] ≡ m
            @test_throws BoundsError m[Inclusion(-1..1)]
            T = Chebyshev(5)
            M = T[m,:]
            @test MemoryLayout(M) isa MappedBasisLayout
            @test MemoryLayout(M[:,1:3]) isa MappedBasisLayout
            @test M[0.1,:] ≈ T[1-2/(0.1+1),:]
            x = axes(M,1)
            @test x == Inclusion(0..Inf)
            @test M \ exp.(-x) ≈ T \ exp.(-(2 ./ (1 .- axes(T,1)) .- 1))

            f = M/M\(1 .- exp.(-x))
            @test f[0.1] ≈ 1 - exp(-0.1) atol=1E-2

            @test f[searchsortedfirst(f, 0.5)] ≈ 0.5

            M = T[InfMap(-1),:]
            @test axes(M,1) == Inclusion(-Inf .. 0)
            x = axes(M,1)
            f = M/M\(exp.(x))
            @test f[-0.1] ≈ exp(-0.1) atol=1E-2
            @test f[searchsortedfirst(f, 0.5)] ≈ 0.5

            M = T[BiInfMap(),:]
            @test axes(M,1) == Inclusion(-Inf .. Inf)
            x = axes(M,1)
            f = M/M\(atan.(x))
            @test f[-0.1] ≈ atan(-0.1) atol=1E-2
            @test f[0] ≈ 0 atol=1E-10
            @test f[searchsortedfirst(f, 0.5)] ≈ 0.5
        end
    end

    @testset "Broadcasted" begin
        T = Chebyshev(5)
        F = factorize(T)
        x = axes(T,1)
        a = 1 .+ x .+ x.^2
        # The following are wrong, just testing dispatch
        @test T \ (a .* T) == I
        @test T \ (a .* (T * (T \ a))) isa Vector
        f = exp.(x) .* a # another broadcast layout
        @test T \ f == F \ f

        ã = T * (T \ a)
        @test T \ (ã .* ã) ≈ [1.5,1,0.5,0,0]

        @test T'*(a .* T) isa Matrix
        @test T'*(a .* (T * (T \ a))) isa Vector
        @test_broken T'f isa Vector
        @test T'ã isa Vector
        @test T'*(ã .* ã) isa Vector
        @test (2T)'*(a .* T) isa Matrix
        @test T'*(2T) isa Matrix
        @test T'*(2T*randn(5)) isa Vector
        @test (2T)'*(T*(1:5)) ≈ T'*(2T*(1:5)) ≈ T'BroadcastQuasiMatrix(*, 2, T*(1:5))
        @test T' * (a .* (T * (1:5))) ≈ T' * ((a .* T) * (1:5))
        @test T'BroadcastQuasiMatrix(*, 2, 2T) == 4*(T'T)

        @test LazyArrays.simplifiable(*, T', T*(1:5)) == Val(true)
        @test LazyArrays.simplifiable(*, T', (a .* (T * (1:5)))) == Val(true)
        @test LazyArrays.simplifiable(*, T', a .* T) == Val(true)
        @test LazyArrays.simplifiable(*, T', 2T) == Val(true)
        @test LazyArrays.simplifiable(*, T', BroadcastQuasiMatrix(*, 2, T*(1:5))) == Val(true)
        @test LazyArrays.simplifiable(*, T', BroadcastQuasiMatrix(*, 2, 2T)) == Val(true)
    end

    @testset "sum/dot/diff" begin
        @test sum(x) ≡ 2.0
        @test sum(exp.(x)) ≈ ℯ - 1/ℯ
        @test dot(x, x) ≈ 2/3
        @test dot(exp.(x), x) ≈ 2/ℯ
        @test diff(exp.(x))[0.1] ≈ exp(0.1)

        @test sum(exp.(x .* (1:2)'); dims=1) ≈ [ℯ - 1/ℯ (ℯ^2 - 1/ℯ^2)/2]
        @test diff(exp.(x .* (1:2)'); dims=1)[0.1,:] ≈ [exp(0.1), exp(0.2)]

        @test_throws ErrorException diff(wT[:,1:3])
        @test_throws ErrorException sum(wT[:,1:3]; dims=1)
        @test_throws ErrorException cumsum(x)
    end

    @testset "Expansion * Lazy" begin
        f = T * collect(1.0:5)
        @test (f * ones(1,4))[0.1,:] == fill(f[0.1],4)
        @test (f * BroadcastArray(exp, (1:4)'))[0.1,:] ≈ f[0.1] * exp.(1:4)
    end

    @testset "undefined domain" begin
        @test_throws ErrorException basis(Inclusion(FooDomain()))
    end

    @testset "Adjoint*Basis not defined" begin
        @test_throws ErrorException Chebyshev(5)'LinearSpline([-1,1])
        @test_throws ErrorException FooBasis()'FooBasis()
    end
end
