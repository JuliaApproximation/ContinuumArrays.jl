

struct DiracDelta{T,A} <: AbstractQuasiVector{T}
    x::T
    axis::A
    DiracDelta{T,A}(x, axis) where {T,A} = new{T,A}(x,axis)
end

DiracDelta{T}(x, axis::A) where {T,A<:AbstractQuasiVector} = DiracDelta{T,A}(x, axis)
DiracDelta{T}(x, domain) where T = DiracDelta{T}(x, Inclusion(domain))
DiracDelta(x, domain) = DiracDelta{eltype(x)}(x, Inclusion(domain))
DiracDelta(axis::AbstractQuasiVector) = DiracDelta(zero(float(eltype(axis))), axis)
DiracDelta(domain) = DiracDelta(Inclusion(domain))

axes(δ::DiracDelta) = (δ.axis,)
IndexStyle(::Type{<:DiracDelta}) = IndexLinear()

==(a::DiracDelta, b::DiracDelta) = a.axis == b.axis && a.x == b.x

function getindex(δ::DiracDelta{T}, x::Real) where T
    x ∈ δ.axis || throw(BoundsError())
    x == δ.x ? inv(zero(T)) : zero(T)
end


function materialize(M::QMul2{<:QuasiAdjoint{<:Any,<:DiracDelta},<:AbstractQuasiVector})
    A, B = M.args
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    B[parent(A).x]
end

function materialize(M::QMul2{<:QuasiAdjoint{<:Any,<:DiracDelta},<:AbstractQuasiMatrix})
    A, B = M.args
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    B[parent(A).x,:]
end

struct Derivative{T,D} <: AbstractQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Derivative{T}(axis::A) where {T,A<:Inclusion} = Derivative{T,A}(axis)
Derivative{T}(domain) where T = Derivative{T}(Inclusion(domain))
Derivative(axis) = Derivative{Float64}(axis)

axes(D::Derivative) = (D.axis, D.axis)
==(a::Derivative, b::Derivative) = a.axis == b.axis


function materialize(M::QMul2{<:Derivative,<:SubQuasiArray})
    A, B = M.args
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    P = parent(B)
    (Derivative(axes(P,1))*P)[parentindices(B)...]
end


# struct Multiplication{T,F,A} <: AbstractQuasiMatrix{T}
#     f::F
#     axis::A
# end


const Identity{T,D} = QuasiDiagonal{T,Inclusion{T,D}}
