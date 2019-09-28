abstract type Basis{T} <: LazyQuasiMatrix{T} end


const WeightedBasis{T, A<:AbstractQuasiVector, B<:Basis} = BroadcastQuasiMatrix{T,typeof(*),<:Tuple{A,B}}

MemoryLayout(::Type{<:Basis}) = LazyLayout()
ApplyStyle(::typeof(pinv), ::Type{<:Basis}) = LazyQuasiArrayApplyStyle()
pinv(J::Basis) = apply(pinv,J)

_multup(a::Tuple) = Mul(a...)
_multup(a) = a


==(A::Basis, B::Basis) = axes(A) ≠ axes(B) ||
    throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))

ApplyStyle(::typeof(\), ::Type{<:Basis}, ::Type{<:AbstractQuasiMatrix}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:Basis}, ::Type{<:AbstractQuasiVector}) = LdivApplyStyle()

function copy(P::Ldiv{<:Any,<:Any,<:Basis,<:Basis})
    A, B = P.A, P.B
    A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end


include("splines.jl")
