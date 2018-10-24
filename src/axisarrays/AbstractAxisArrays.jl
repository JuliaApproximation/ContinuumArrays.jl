module AbstractAxisArrays
using Base, LinearAlgebra, LazyArrays
import Base: getindex, size, axes, length, ==, isequal, iterate, CartesianIndices, LinearIndices,
                Indices, IndexStyle, getindex, setindex!, parent, vec, convert, similar, zero,
                map, eachindex, eltype
import Base: @_inline_meta, DimOrInd, OneTo, @_propagate_inbounds_meta, @_noinline_meta,
                DimsInteger, error_if_canonical_getindex, @propagate_inbounds, _return_type, _default_type,
                _maybetail, tail, _getindex, _maybe_reshape, index_ndims, _unsafe_getindex,
                index_shape, to_shape, unsafe_length, @nloops, @ncall
import Base: ViewIndex, Slice, ScalarIndex, RangeIndex
import Base: *, /, \, +, -, inv

import LinearAlgebra: transpose, adjoint, checkeltype_adjoint, checkeltype_transpose

import LazyArrays: MemoryLayout, UnknownLayout

export AbstractAxisArray, AbstractAxisMatrix, AbstractAxisVector, materialize

abstract type AbstractAxisArray{T,N} end
AbstractAxisVector{T} = AbstractAxisArray{T,1}
AbstractAxisMatrix{T} = AbstractAxisArray{T,2}
AbstractAxisVecOrMat{T} = Union{AbstractAxisVector{T}, AbstractAxisMatrix{T}}


_length(d) = length(d)

size(A::AbstractAxisArray) = _length.(axes(A))
axes(A::AbstractAxisArray) = error("Override axes for $(typeof(A))")

include("indices.jl")
include("abstractaxisarray.jl")
include("multidimensional.jl")
include("subarray.jl")
include("matmul.jl")

include("adjtrans.jl")

end
