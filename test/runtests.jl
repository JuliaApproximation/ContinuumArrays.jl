using ContinuumArrays, QuasiArrays, IntervalSets, DomainSets, FillArrays, LinearAlgebra, BandedMatrices, InfiniteArrays, Test, Base64
import ContinuumArrays: ℵ₁, materialize, AffineQuasiVector, BasisLayout, AdjointBasisLayout, SubBasisLayout, ℵ₁,
                        MappedBasisLayout, AdjointMappedBasisLayouts, MappedWeightedBasisLayout, TransformFactorization, Weight, WeightedBasisLayout, SubWeightedBasisLayout, WeightLayout,
                        basis, invmap, Map, checkpoints, plotgrid, plotgrid_layout, mul, plotvalues
import QuasiArrays: SubQuasiArray, MulQuasiMatrix, Vec, Inclusion, QuasiDiagonal, LazyQuasiArrayApplyStyle, LazyQuasiArrayStyle
import LazyArrays: MemoryLayout, ApplyStyle, Applied, colsupport, arguments, ApplyLayout, LdivStyle, MulStyle


@testset "Inclusion" begin
    @testset "basics" begin
        x = Inclusion(-1..1)
        @test eltype(x) == Float64
        @test x[0.1] ≡ 0.1
        @test x[0] ≡ x[0.0] ≡ 0.0

        @test size(Inclusion(ℝ),1) == ℵ₁
        @test size(Inclusion(ℤ),1) == ℵ₀
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
        @test stringmime("text/plain", D) == "Derivative(Inclusion($(-1..1)))"
        @test_throws DimensionMismatch Derivative(Inclusion(0..1)) * x
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

    @test stringmime("text/plain", δ) == "δ at 0.0 over Inclusion($(-1..3))"
    x = Inclusion(-1..1)
    @test stringmime("text/plain", δ[2x .+ 1]) == "δ at 0.0 over Inclusion($(-1..3)) affine mapped to $(-1..1)"
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

@testset "Grids/values" begin
    L = LinearSpline(1:5)
    c = randn(5)
    u = L*c
    @test plotvalues(u) == u[plotgrid(u)]

    a = affine(0..1, 1..5)
    v = L[a,:] * c
    @test plotvalues(v) == v[plotgrid(v)]
end

include("test_recipesbaseext.jl")
include("test_makieext.jl")
