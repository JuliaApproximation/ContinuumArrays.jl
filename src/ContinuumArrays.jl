module ContinuumArrays
using Base, LinearAlgebra, IntervalSets
import Base: getindex, size, axes, length, ==, isequal, iterate, CartesianIndices, LinearIndices,
                Indices, IndexStyle, getindex, setindex!, parent, vec, convert, similar, zero,
                map, eachindex
import Base: @_inline_meta, DimOrInd, OneTo, @_propagate_inbounds_meta, @_noinline_meta,
                DimsInteger, error_if_canonical_getindex, @propagate_inbounds, _return_type, _default_type,
                _maybetail, tail, _getindex, _maybe_reshape, index_ndims, _unsafe_getindex,
                index_shape, to_shape, unsafe_length, @nloops, @ncall


import LinearAlgebra: transpose, adjoint

abstract type AbstractAxisArray{T,N} end
AbstractAxisVector{T} = AbstractAxisArray{T,1}
AbstractAxisMatrix{T} = AbstractAxisArray{T,2}
AbstractAxisVecOrMat{T} = Union{AbstractAxisVector{T}, AbstractAxisMatrix{T}}

struct ℵ₀ <: Number end
_length(::AbstractInterval) = ℵ₀
_length(d) = length(d)

size(A::AbstractAxisArray) = _length.(axes(A))
axes(A::AbstractAxisArray) = error("Override axes for $(typeof(A))")

include("indices.jl")
include("abstractaxisarray.jl")
include("adjtrans.jl")
include("multidimensional.jl")
end
