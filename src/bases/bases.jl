abstract type Basis{T} <: AbstractQuasiMatrix{T} end
struct BasisStyle <: ApplyStyle end

for op in (:*, :pinv, :inv)
    @eval ApplyStyle(::typeof($op), ::Basis) = BasisStyle()
end
ApplyStyle(_, ::Basis) = BasisStyle()
ApplyStyle(_, ::Basis, ::AbstractArray) = BasisStyle()
ApplyStyle(::typeof(*), ::Basis, ::AbstractArray) = BasisStyle()


pinv(J::Basis) = apply(pinv,J)
materialize(P::Applied{BasisStyle,typeof(pinv)}) = ApplyQuasiMatrix(pinv,parent(P))

_multup(a::Tuple) = Mul(a...)
_multup(a) = a
function materialize(M::Mul{<:Any,<:Tuple{<:PInv{<:Any,<:Basis},Vararg{Any}}}) 
    a,b = M.args
    apply(\,pinv(a),_multup(b))
end

materialize(M::Mul{BasisStyle,<:Tuple{<:Basis,<:AbstractArray}}) = ApplyQuasiArray(M)


ApplyStyle(::typeof(\), ::Basis, ::Basis) =
    BasisStyle()

ApplyStyle(::typeof(\), ::Basis, ::Applied) = 
    BasisStyle()
ApplyStyle(::typeof(\), A::Basis, B::ApplyQuasiArray) = ApplyStyle(\, A, B.applied)

==(A::Basis, B::Basis) = axes(A) ≠ axes(B) ||
    throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))

materialize(P::Applied{BasisStyle,typeof(\)}) = error("Implement")

function materialize(P::Applied{BasisStyle,typeof(\),<:Tuple{<:Basis,<:Basis}})
    A, B = P.args
    axes(A) == axes(B) || throw(DimensionMismatch("axes of bases must match"))
    A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end

function materialize(P::Applied{BasisStyle,typeof(\),<:Tuple{<:Basis,<:Mul}})
    A, B = P.args
    *(A \ first(B.args), tail(B.args)...)
end

function materialize(P::Applied{BasisStyle,typeof(\),<:Tuple{<:Basis,<:ApplyQuasiArray}})
    A, B = P.args
    apply(\, A, B.applied)
end


include("splines.jl")
include("orthogonalpolynomials.jl")
include("jacobi.jl")
include("ultraspherical.jl")
