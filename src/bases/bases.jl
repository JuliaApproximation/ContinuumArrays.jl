abstract type Basis{T} <: AbstractQuasiMatrix{T} end
struct BasisStyle <: ApplyStyle end

for op in (:*, :pinv, :inv)
    @eval ApplyStyle(::typeof($op), ::Basis) = BasisStyle()
end
ApplyStyle(_, ::Basis) = BasisStyle()

pinv(J::Basis) = materialize(PInv(J))
materialize(P::Applied{BasisStyle,typeof(pinv)}) = ApplyQuasiMatrix(pinv,parent(P))

ApplyStyle(::typeof(\), ::Basis, ::Basis) =
    BasisStyle()

==(A::Basis, B::Basis) = axes(A) ≠ axes(B) ||
    throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))

function materialize(P::Applied{BasisStyle,typeof(\)})
    Ai, B = P.args
    A = parent(Ai)
    axes(A) == axes(B) || throw(DimensionMismatch("axes of bases must match"))
    A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end


include("splines.jl")
include("jacobi.jl")
