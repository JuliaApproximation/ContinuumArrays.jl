struct JacobiWeight{T} <: AbstractQuasiVector{T}
    b::T
    a::T
end

axes(::JacobiWeight) = (ChebyshevInterval(),)
function getindex(w::JacobiWeight, x::Real)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x)^w.a * (1+x)^w.b
end

abstract type AbstractJacobi{T} <: AbstractQuasiMatrix{T} end

struct Legendre{T} <: AbstractJacobi{T} end
Legendre() = Legendre{Float64}()

struct Jacobi{T} <: AbstractJacobi{T}
    b::T
    a::T
end

axes(::AbstractJacobi) = (ChebyshevInterval(), OneTo(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b


materialize(M::Mul2{<:Any,<:Any,<:QuasiAdjoint{<:Any,<:Legendre},<:Legendre}) =
    Diagonal(2 ./ (2(0:∞) .+ 1))


function materialize(M::Mul2{<:Any,<:Any,<:Derivative{<:Any,<:ChebyshevInterval},<:Jacobi})
    _, S = M.factors
    D = _BandedMatrix(((1:∞ .+ 1 .+ S.a .+ S.b)/2)', ∞, -1,1)
    Mul(Jacobi(S.a+1,S.b+1), D)
end

function materialize(M::Mul{<:Tuple,<:Tuple{<:Derivative{<:Any,<:ChebyshevInterval},
                                        QuasiDiagonal{Bool,JacobiWeight{Bool}},Jacobi{Bool}}})
    _, W, S = M.factors
    w = parent(W)
    (w.a && S.a && w.b && S.b) || throw(ArgumentError())
    D = _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
    Mul(Legendre(), D)
end
