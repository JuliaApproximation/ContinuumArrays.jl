abstract type Basis{T} <: LazyQuasiMatrix{T} end
struct BasisLayout <: MemoryLayout end


MemoryLayout(::Type{<:Basis}) = BasisLayout()



pinv(J::Basis) = apply(pinv,J)

_multup(a::Tuple) = Mul(a...)
_multup(a) = a


==(A::Basis, B::Basis) = axes(A) ≠ axes(B) ||
    throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))


function materialize(P::Ldiv{BasisLayout,BasisLayout})
    A, B = P.args
    axes(A) == axes(B) || throw(DimensionMismatch("axes of bases must match"))
    A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end


include("splines.jl")
include("orthogonalpolynomials.jl")
include("jacobi.jl")
include("ultraspherical.jl")
