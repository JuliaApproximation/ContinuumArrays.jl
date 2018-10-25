module QuasiArrays
using Base, LinearAlgebra, LazyArrays
import Base: getindex, size, axes, length, ==, isequal, iterate, CartesianIndices, LinearIndices,
                Indices, IndexStyle, getindex, setindex!, parent, vec, convert, similar, zero,
                map, eachindex, eltype, first, last
import Base: @_inline_meta, DimOrInd, OneTo, @_propagate_inbounds_meta, @_noinline_meta,
                DimsInteger, error_if_canonical_getindex, @propagate_inbounds, _return_type, _default_type,
                _maybetail, tail, _getindex, _maybe_reshape, index_ndims, _unsafe_getindex,
                index_shape, to_shape, unsafe_length, @nloops, @ncall, Slice
import Base: ViewIndex, Slice, ScalarIndex, RangeIndex
import Base: *, /, \, +, -, inv

import Base.Broadcast: materialize

import LinearAlgebra: transpose, adjoint, checkeltype_adjoint, checkeltype_transpose

import LazyArrays: MemoryLayout, UnknownLayout, Mul2

export AbstractQuasiArray, AbstractQuasiMatrix, AbstractQuasiVector, materialize

abstract type AbstractQuasiArray{T,N} end
AbstractQuasiVector{T} = AbstractQuasiArray{T,1}
AbstractQuasiMatrix{T} = AbstractQuasiArray{T,2}
AbstractQuasiVecOrMat{T} = Union{AbstractQuasiVector{T}, AbstractQuasiMatrix{T}}


_length(d) = length(d)

size(A::AbstractQuasiArray) = _length.(axes(A))
axes(A::AbstractQuasiArray) = error("Override axes for $(typeof(A))")

include("indices.jl")
include("abstractquasiarray.jl")
include("multidimensional.jl")
include("subarray.jl")
include("matmul.jl")

include("adjtrans.jl")

end
