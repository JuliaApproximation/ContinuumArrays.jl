

struct DiracDelta{T,A} <: AbstractQuasiVector{T}
    x::T
    axis::A
end

DiracDelta{T}(x, axis::A) where {T,A} = DiracDelta{T,A}(x, axis)
DiracDelta(axis) = DiracDelta(zero(float(eltype(axis))), axis)

axes(δ::DiracDelta) = (δ.axis,)
IndexStyle(::Type{<:DiracDelta}) = IndexLinear()

==(a::DiracDelta, b::DiracDelta) = a.axis == b.axis && a.x == b.x

function getindex(δ::DiracDelta{T}, x::Real) where T
    x ∈ δ.axis || throw(BoundsError())
    x == δ.x ? inv(zero(T)) : zero(T)
end


function materialize(M::Mul2{<:Any,<:Any,<:QuasiArrays.Adjoint{<:Any,<:DiracDelta},<:AbstractQuasiVector})
    A, B = M.A, M.B
    axes(A,2) == axes(A,1) || throw(DimensionMismatch())
    B[parent(A).x]
end

function materialize(M::Mul2{<:Any,<:Any,<:QuasiArrays.Adjoint{<:Any,<:DiracDelta},<:AbstractQuasiMatrix})
    A, B = M.factors
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    B[parent(A).x,:]
end

struct Derivative{T,A} <: AbstractQuasiVector{T}
    axis::A
end

Derivative{T}(axis::A) where {T,A} = Derivative{T,A}(axis)
Derivative(axis) = Derivative{Float64}(axis)

axes(D::Derivative) = (D.axis, D.axis)
==(a::Derivative, b::Derivative) = a.axis == b.axis
