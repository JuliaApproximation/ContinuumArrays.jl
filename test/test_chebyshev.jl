using ContinuumArrays, LinearAlgebra, FastTransforms, Test
import ContinuumArrays: Basis, Weight, Map, LazyQuasiArrayStyle, TransformFactorization

"""
This is a simple implementation of Chebyshev for testing. Use ClassicalOrthogonalPolynomials
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
LinearAlgebra.factorize(L::Chebyshev, n) =
    TransformFactorization(grid(L), plan_chebyshevtransform(Array{Float64}(undef, size(L,2),n),1))    

# This is wrong but just for tests
QuasiArrays.layout_broadcasted(::LazyQuasiArrayStyle{2}, ::Tuple{ExpansionLayout,Any}, ::typeof(*), a::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{Chebyshev,Any}}, b::Chebyshev) = b * Matrix(I, 5, 5)

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

        @test T \ [exp.(x) cos.(x)] ≈ [F \ exp.(x) F \ cos.(x)]
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

        @test wT \ w ≈ [1; zeros(4)]
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
        @test T \ (a .* (T * (T \ a))) isa Vector
        f = exp.(x) .* a # another broadcast layout
        @test T \ f == F \ f

        ã = T * (T \ a)
        @test T \ (ã .* ã) ≈ [1.5,1,0.5,0,0]
    end
end