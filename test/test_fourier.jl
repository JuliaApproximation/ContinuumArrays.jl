using ContinuumArrays, LazyArrays, Test
import ContinuumArrays: Basis

struct Fourier <: Basis{ComplexF64}
    n::Int
end

struct CosBasis <: Basis{Float64}
    n::Int
end

struct SinBasis <: Basis{Float64}
    n::Int
end

Base.isreal(::Fourier) = false
Base.real(F::Fourier) = CosBasis(F.n)
Base.imag(F::Fourier) = [zeros(axes(F,1)) SinBasis(F.n-1)]

Base.getindex(::Fourier, θ::Real, j::Int) = exp(im*(j-1)*θ)
Base.getindex(::CosBasis, θ::Real, j::Int) = cos((j-1)*θ)
Base.getindex(::SinBasis, θ::Real, j::Int) = sin(j*θ)

Base.axes(F::Union{Fourier,CosBasis,SinBasis}) = (Inclusion(0..2π), Base.OneTo(F.n))
@testset "Fourier" begin
    @testset "reim" begin
        F = Fourier(5)
        f = F * (collect(1:5) .+ im)
        @test real(f)[0.1] ≈ real(f[0.1])
        @test imag(f)[0.1] ≈ imag(f[0.1])
    end
end