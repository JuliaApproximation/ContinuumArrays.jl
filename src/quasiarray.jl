import Base: *

abstract type QuasiArray{N,T} end

QuasiVector{T} = QuasiArray{1,T}

struct DiracDelta{T}
    x   ::  T
end

struct Fun{Q <: QuasiArray,C <: AbstractArray}
    q_array         ::  Q
    coefficients    ::  C
end

# Note: this will allocate memory for a vector, which could be avoided.
# On the other hand, evaluating the whole quasivector in one go may be more
# efficient than evaluating the elements one by one
(*)(δ::DiracDelta, f::Fun) = (δ*f.q_array) * f.coefficients

(*)(a::QuasiArray, c::AbstractArray) = Fun(a, c)

(f::Fun)(x) = DiracDelta(x) * f


# This is a special type of coefficient vector that singles out one specific
# basis function from a set.
struct UnitVector <: AbstractVector{Int}
    # Should it have a length? Perhaps there should be an infinite and a finite version
    # Perhaps even with index k::I, where I can be any index that the quasiarray supports
    k   ::  Int
end

UnitFun{Q <: QuasiArray} = Fun{Q,UnitVector}

Base.getindex(v::UnitVector, i::Int) = (i == v.k) ? 1 : 0

# Index into a quasivector and you get a function
Base.getindex(a::QuasiVector, k::Int) = a * UnitVector(k)

# Perhaps there is a special case to evaluate individual basis functions?
# (*)(δ::DiracDelta, f::UnitFun) = ...


# Legendre is an infinite vector of functions
struct Legendre{T} <: QuasiVector{T}
end

size(b::Legendre) = (1,∞)

eachindex(b::Legendre) = 1:∞

struct LazyLegendreEvaluation{T}
    x   ::  T
end

(*)(δ::DiracDelta, b::Legendre) = LazyLegendreEvaluation(x)

# implement infinite multiplication here
# (*)(a::LazyLegendreEvaluation, a::InfiniteArray) = ...



# This is a finite Fourier series with frequencies from -n to n
struct FiniteFourier{T} <: QuasiVector{T}
    n   ::  Int
end

length(b::FiniteFourier) = 2b.n+1

size(b::FiniteFourier) = (1,length(b))

# Evaluate the functions exp(2*pi*i*k*x)
(*)(δ::DiracDelta, b::FiniteFourier{T}) where {T} = [exp(2*convert(T, pi)*im*k*δ.x) for k in -b.length:b.length]'

zeros(q::QuasiVector) = Fun(q, zeros(length(q)))
