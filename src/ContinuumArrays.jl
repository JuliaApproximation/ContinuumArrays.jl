module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, BandedMatrices, InfiniteArrays, DomainSets
import Base: @_inline_meta, axes, getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail
import Base.Broadcast: materialize
import LazyArrays: Mul2, MemoryLayout, Applied
import LinearAlgebra: pinv
import BandedMatrices: AbstractBandedLayout, _BandedMatrix

include("QuasiArrays/QuasiArrays.jl")
using .QuasiArrays
import .QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, slice, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, JacobiWeight, Jacobi, Legendre,
            fullmaterialize

####
# Interval indexing support
####
struct AlephInfinity{N} <: Integer end

const ℵ₁ = AlephInfinity{1}()


const QMul2{A,B} = Mul{<:Any, <:Tuple{A,B}}

cardinality(::AbstractInterval) = ℵ₁
*(ℵ::AlephInfinity) = ℵ


checkindex(::Type{Bool}, inds::AbstractInterval, i::Real) = (leftendpoint(inds) <= i) & (i <= rightendpoint(inds))
checkindex(::Type{Bool}, inds::AbstractInterval, i::Inclusion) = i.axis ⊆ inds
function checkindex(::Type{Bool}, inds::AbstractInterval, I::AbstractArray)
    @_inline_meta
    b = true
    for i in I
        b &= checkindex(Bool, inds, i)
    end
    b
end


# we represent as a Mul with a banded matrix
function materialize(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractUnitRange}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = _BandedMatrix(Ones{Int}(1,length(jr)), axes(A,2), first(jr)-1,1-first(jr))
    A*P
end




most(a) = reverse(tail(reverse(a)))

MulQuasiOrArray = Union{MulArray,MulQuasiArray}

_factors(M::MulQuasiOrArray) = M.applied.args
_factors(M) = (M,)

_flatten() = ()
_flatten(A, B...) = (A, _flatten(B...)...)
_flatten(A::Mul, B...) = _flatten(A.args..., B...)
flatten(A::Mul) = Mul(_flatten(A.args...)...)

_flatten(A::MulQuasiArray, B...) = _flatten(A.applied, B...)
flatten(A::MulQuasiArray) = MulQuasiArray(flatten(A.applied))

function fullmaterialize(M::Applied{<:Any,typeof(*)})
    M_mat = materialize(flatten(M))
    typeof(M_mat) <: MulQuasiOrArray || return M_mat
    typeof(M_mat.applied) == typeof(M) || return(fullmaterialize(M_mat))

    ABC = M_mat.applied.args
    length(ABC) ≤ 2 && return M_mat

    AB = most(ABC)
    Mhead = fullmaterialize(Mul(AB...))

    typeof(_factors(Mhead)) == typeof(AB) ||
        return fullmaterialize(Mul(_factors(Mhead)..., last(ABC)))

    BC = tail(ABC)
    Mtail =  fullmaterialize(Mul(BC...))
    typeof(_factors(Mtail)) == typeof(BC) ||
        return fullmaterialize(Mul(first(ABC), _factors(Mtail)...))

    first(ABC) * Mtail
end

fullmaterialize(M::ApplyQuasiArray) = fullmaterialize(M.applied)
fullmaterialize(M) = M

materialize(M::Applied{<:Any,typeof(*),<:Tuple{Vararg{<:Union{Adjoint,QuasiAdjoint,QuasiDiagonal}}}}) =
    materialize(Mul(reverse(adjoint.(M.args))...))'

include("operators.jl")
include("bases/bases.jl")

end
