using ContinuumArrays, LinearAlgebra, Test

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
        @test summary(f) == "(HeavisideSpline{Float64, Vector{$Int}}) * (2-element Vector{$Int})"
        @test stringmime("text/plain", f) == "HeavisideSpline{Float64, Vector{$Int}} * [1, 2]"
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
            @test δ'f == mul(δ',f) == dot(δ,f) == f[1.2]

            @test @inferred(L'L) == SymTridiagonal([1/3,2/3,1/3], [1/6,1/6])

            @testset "Algebra" begin
                @test 2f == f*2 == 2 .* f == f .* 2
                @test 2\f == f/2 == 2 .\ f == f ./ 2
                @test sum(f) ≈ 4.5
                @test L \ BroadcastQuasiArray(*,2,f) ≈ [2,4,8]
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

            @test L \ zeros(x) ≡ Zeros(10_000)
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

            @test L[y,:] \ (y .* exp.(y)) ≈ L[y,:] \ BroadcastQuasiVector(y -> y*exp(y), y)
            @test L[y,:] \ (y .* L[y,1:3]) ≈ [L[y,:]\(y .* L[y,1]) L[y,:]\(y .* L[y,2]) L[y,:]\(y .* L[y,3])]

            c = randn(size(L,2))
            @test L[y,:] \ (L[y,:] * c) ≈ c
            @test ContinuumArrays.demap(L[y,:] * c) == L*c
        end

        @testset "Mapped and BroadcastLayout{typeof(+)}" begin
            @test L[y,:] \ (y .+ y) ≈ L[y,:] \ (2y)
            @test L[y,:] \ (y .- y) ≈ zeros(10)
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
        @test stringmime("text/plain", B) == "HeavisideSpline{Float64, Vector{$Int}} affine mapped to 0..1"
    end

    @testset "A \\ ( c .* B) == c .* (A\\B) #101" begin
        L = LinearSpline(0:5)
        @test L \ (2L) == 2(L\L)
    end

    @testset "H\\D*L" begin
        L = LinearSpline(0:5)
        H = HeavisideSpline(L)
        x = axes(L,1)
        D = Derivative(x)
        @test H\D*L == H\(D*L)
    end
end