abstract type Basis{T} <: AbstractQuasiMatrix{T} end
struct BasisLayout <: MemoryLayout end

MemoryLayout(::Basis) = BasisLayout()

pinv(J::Basis) = materialize(PInv(J))
materialize(P::PInv{BasisLayout}) = _PInvQuasiMatrix(P)

==(A::Basis, B::Basis) = axes(A) ≠ axes(B) ||
    throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))

function materialize(P::Ldiv{BasisLayout,BasisLayout})
    Ai, B = P.args
    A = parent(Ai)
    axes(A) == axes(B) || throw(DimensionMismatch("axes of bases must match"))
    A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end

include("splines.jl")
include("jacobi.jl")
