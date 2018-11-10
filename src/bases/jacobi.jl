struct JacobiWeight{T} <: AbstractQuasiVector{T}
    b::T
    a::T
end

axes(::JacobiWeight) = (ChebyshevInterval(),)
function getindex(w::JacobiWeight, x::Real)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x)^w.a * (1+x)^w.b
end


struct Jacobi{T} <: AbstractQuasiMatrix{T}
    b::T
    a::T
end

axes(::Jacobi) = (ChebyshevInterval(), OneTo(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b
