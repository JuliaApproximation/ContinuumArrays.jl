module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, BandedMatrices, InfiniteArrays, DomainSets
import Base: @_inline_meta, axes, getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail
import Base.Broadcast: materialize
import LazyArrays: Mul2, MemoryLayout, Applied, ApplyStyle
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
checkindex(::Type{Bool}, inds::AbstractInterval, i::Inclusion) = i.domain ⊆ inds
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



include("operators.jl")
include("bases/bases.jl")

end
