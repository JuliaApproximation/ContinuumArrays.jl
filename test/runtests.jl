using ContinuumArrays, QuasiArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, BandedMatrices, InfiniteArrays, Test, Base64, RecipesBase
import ContinuumArrays: ℵ₁, materialize, AffineQuasiVector, BasisLayout, AdjointBasisLayout, SubBasisLayout, ℵ₁,
                        MappedBasisLayout, AdjointMappedBasisLayout, MappedWeightedBasisLayout, TransformFactorization, Weight, WeightedBasisLayout, SubWeightedBasisLayout, WeightLayout,
                        Expansion, basis, invmap, Map, checkpoints, _plotgrid, mul
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
        @test D == Derivative{Float64}(x) == Derivative{Float64}(-1..1)
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

@testset "Kernels" begin
    x = Inclusion(0..1)
    K = x .- x'
    @test K[0.1,0.2] == K[Inclusion(0..0.5), Inclusion(0..0.5)][0.1,0.2] == 0.1 - 0.2
    @test_throws BoundsError K[Inclusion(0..0.5), Inclusion(0..0.5)][1,1]

    L = LinearSpline(0:0.1:1)
    @test_broken L\K isa ApplyQuasiArray # broken since it assumes non-broadcasting
    @test L \ exp.(K) isa ApplyQuasiArray
    
end

include("test_splines.jl")
include("test_chebyshev.jl")
include("test_basisconcat.jl")

@testset "Plotting" begin
    L = LinearSpline(0:5)
    rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), L)
    @test rep[1].args == (L.points,L[L.points,:])

    u = L*randn(6)
    rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
    @test rep[1].args == (L.points,u[L.points])

    @testset "padded" begin
        u = L * Vcat(rand(3), Zeros(3))
        rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
        @test rep[1].args == (L.points,u[L.points])
    end

    @testset "Chebyshev and weighted Chebyshev" begin
        T =  Chebyshev(10)
        w =  ChebyshevWeight()
        wT = w .* T
        x =  axes(T, 1)
    
        u = T * Vcat(rand(3), Zeros(7))
        v = wT * Vcat(rand(3), Zeros(7))
    
        rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
        @test rep[1].args == (grid(T), u[grid(T)])
        wrep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), v)
        @test wrep[1].args == (grid(wT), v[grid(wT)])
    
        @test plotgrid(v) == plotgrid(u) == grid(T) == grid(wT) == _plotgrid(MemoryLayout(v), v) == _plotgrid(MemoryLayout(u), u)
        y = affine(0..1, x)
        @test plotgrid(T[y,:]) == (plotgrid(T) .+ 1)/2
    end
end