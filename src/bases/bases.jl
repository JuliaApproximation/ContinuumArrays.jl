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
ApplyStyle(::typeof(\), ::Type{<:SubQuasiArray{<:Any,2,<:Basis}}, ::Type{<:AbstractQuasiMatrix}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:SubQuasiArray{<:Any,2,<:Basis}}, ::Type{<:AbstractQuasiVector}) = LdivApplyStyle()

function copy(P::Ldiv{<:Any,<:Any,<:Basis,<:Basis})
    A, B = P.A, P.B
    A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end

function copy(P::Ldiv{<:Any,<:Any,<:SubQuasiArray{<:Any,2,<:Basis},<:SubQuasiArray{<:Any,2,<:Basis}})
    A, B = P.A, P.B
    (parent(A) == parent(B) && parentindices(A) == parentindices(B)) || 
        throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end

## materialize views

# materialize(S::SubQuasiArray{<:Any,2,<:ApplyQuasiArray{<:Any,2,typeof(*),<:Tuple{<:Basis,<:Any}}}) =
#     *(arguments(S)...)


# Differentiation of sub-arrays 
function copy(M::QMul2{<:Derivative,<:SubQuasiArray{<:Any,2,<:Basis,<:Tuple{<:Inclusion,<:Any}}})
    A, B = M.args
    P = parent(B)
    (Derivative(axes(P,1))*P)[parentindices(B)...]
end

function copy(M::QMul2{<:Derivative,<:SubQuasiArray{<:Any,2,<:Basis,<:Tuple{<:AffineMap,<:Any}}})
    A, B = M.args
    P = parent(B)
    kr,jr = parentindices(B)
    (Derivative(axes(P,1))*P*kr.A)[kr,jr]
end


include("splines.jl")

