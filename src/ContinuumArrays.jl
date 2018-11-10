module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, BandedMatrices, InfiniteArrays, DomainSets
import Base: @_inline_meta, axes, getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo
import Base.Broadcast: materialize
import LazyArrays: Mul2
import BandedMatrices: AbstractBandedLayout, _BandedMatrix

include("QuasiArrays/QuasiArrays.jl")
using .QuasiArrays
import .QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, slice, QSlice, SubQuasiArray,
                    QuasiDiagonal

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, JacobiWeight, Jacobi, Legendre

####
# Interval indexing support
####
struct AlephInfinity{N} <: Integer end

const ℵ₁ = AlephInfinity{1}()


cardinality(::AbstractInterval) = ℵ₁
*(ℵ::AlephInfinity) = ℵ


checkindex(::Type{Bool}, inds::AbstractInterval, i::Real) = (leftendpoint(inds) <= i) & (i <= rightendpoint(inds))
checkindex(::Type{Bool}, inds::AbstractInterval, i::QSlice) = i.axis ⊆ inds
function checkindex(::Type{Bool}, inds::AbstractInterval, I::AbstractArray)
    @_inline_meta
    b = true
    for i in I
        b &= checkindex(Bool, inds, i)
    end
    b
end


# we represent as a Mul with a banded matrix
function materialize(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:QSlice,<:AbstractUnitRange}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = BandedMatrix((1-first(jr) => Ones{Int}(length(jr)),), (size(A,2), length(jr)))
    A*P
end


include("operators.jl")
include("bases/bases.jl")

end
