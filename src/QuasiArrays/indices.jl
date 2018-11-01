# This file is a part of Julia. License is MIT: https://julialang.org/license


IndexStyle(A::AbstractQuasiArray) = IndexStyle(typeof(A))
IndexStyle(::Type{<:AbstractQuasiArray}) = IndexCartesian()

IndexStyle(A::AbstractQuasiArray, B::AbstractQuasiArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractArray, B::AbstractQuasiArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractQuasiArray, B::AbstractArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
IndexStyle(A::AbstractQuasiArray, B::AbstractQuasiArray...) = IndexStyle(IndexStyle(A), IndexStyle(B...))
IndexStyle(A::AbstractQuasiArray, B::AbstractArray...) = IndexStyle(IndexStyle(A), IndexStyle(B...))

function promote_shape(a::AbstractQuasiArray, b::AbstractQuasiArray)
    promote_shape(axes(a), axes(b))
end


# convert to a supported index type (array or Int)
"""
    to_index(A, i)

Convert index `i` to an `Int` or array of indices to be used as an index into array `A`.

Custom array types may specialize `to_index(::CustomArray, i)` to provide
special indexing behaviors. Note that some index types (like `Colon`) require
more context in order to transform them into an array of indices; those get
converted in the more complicated `to_indices` function. By default, this
simply calls the generic `to_index(i)`. This must return either an `Int` or an
`AbstractArray` of scalar indices that are supported by `A`.
"""
to_index(A, i) = to_index(i)

"""
    to_index(i)

Convert index `i` to an `Int` or array of `Int`s to be used as an index for all arrays.

Custom index types may specialize `to_index(::CustomIndex)` to provide special
indexing behaviors. This must return either an `Int` or an `AbstractArray` of
`Int`s.
"""
to_index(i::Real) = i
to_index(i::Bool) = throw(ArgumentError("invalid index: $i of type $(typeof(i))"))
to_index(I::AbstractArray{Bool}) = LogicalIndex(I)
to_index(I::AbstractArray) = I
to_index(I::AbstractArray{<:Union{AbstractArray, Colon}}) =
    throw(ArgumentError("invalid index: $I of type $(typeof(I))"))
to_index(::Colon) = throw(ArgumentError("colons must be converted by to_indices(...)"))
to_index(i) = throw(ArgumentError("invalid index: $i of type $(typeof(i))"))

# The general to_indices is mostly defined in multidimensional.jl, but this
# definition is required for bootstrap:
"""
    to_indices(A, I::Tuple)

Convert the tuple `I` to a tuple of indices for use in indexing into array `A`.

The returned tuple must only contain either `Int`s or `AbstractArray`s of
scalar indices that are supported by array `A`. It will error upon encountering
a novel index type that it does not know how to process.

For simple index types, it defers to the unexported `Base.to_index(A, i)` to
process each index `i`. While this internal function is not intended to be
called directly, `Base.to_index` may be extended by custom array or index types
to provide custom indexing behaviors.

More complicated index types may require more context about the dimension into
which they index. To support those cases, `to_indices(A, I)` calls
`to_indices(A, axes(A), I)`, which then recursively walks through both the
given tuple of indices and the dimensional indices of `A` in tandem. As such,
not all index types are guaranteed to propagate to `Base.to_index`.
"""
to_indices(A, I::Tuple) = (@_inline_meta; to_indices(A, axes(A), I))
to_indices(A, I::Tuple{Any}) = (@_inline_meta; to_indices(A, (eachindex(IndexLinear(), A),), I))
to_indices(A, inds, ::Tuple{}) = ()
to_indices(A, inds, I::Tuple{Any, Vararg{Any}}) =
    (@_inline_meta; (to_index(A, I[1]), to_indices(A, _maybetail(inds), tail(I))...))

# check for valid sizes in A[I...] = X where X <: AbstractQuasiArray
# we want to allow dimensions that are equal up to permutation, but only
# for permutations that leave array elements in the same linear order.
# those are the permutations that preserve the order of the non-singleton
# dimensions.
function setindex_shape_check(X::AbstractQuasiArray, I::Integer...)
    li = ndims(X)
    lj = length(I)
    i = j = 1
    while true
        ii = length(axes(X,i))
        jj = I[j]
        if i == li || j == lj
            while i < li
                i += 1
                ii *= length(axes(X,i))
            end
            while j < lj
                j += 1
                jj *= I[j]
            end
            if ii != jj
                throw_setindex_mismatch(X, I)
            end
            return
        end
        if ii == jj
            i += 1
            j += 1
        elseif ii == 1
            i += 1
        elseif jj == 1
            j += 1
        else
            throw_setindex_mismatch(X, I)
        end
    end
end

setindex_shape_check(X::AbstractQuasiArray) =
    (length(X)==1 || throw_setindex_mismatch(X,()))

setindex_shape_check(X::AbstractQuasiArray, i::Integer) =
    (length(X)==i || throw_setindex_mismatch(X, (i,)))

setindex_shape_check(X::AbstractQuasiArray{<:Any,1}, i::Integer) =
    (length(X)==i || throw_setindex_mismatch(X, (i,)))

setindex_shape_check(X::AbstractQuasiArray{<:Any,1}, i::Integer, j::Integer) =
    (length(X)==i*j || throw_setindex_mismatch(X, (i,j)))

function setindex_shape_check(X::AbstractQuasiArray{<:Any,2}, i::Integer, j::Integer)
    if length(X) != i*j
        throw_setindex_mismatch(X, (i,j))
    end
    sx1 = length(axes(X,1))
    if !(i == 1 || i == sx1 || sx1 == 1)
        throw_setindex_mismatch(X, (i,j))
    end
end


to_index(I::AbstractQuasiArray{Bool}) = LogicalIndex(I)
to_index(I::AbstractQuasiArray) = I
to_index(I::AbstractQuasiArray{<:Union{AbstractArray, Colon}}) =
    throw(ArgumentError("invalid index: $I of type $(typeof(I))"))

LinearIndices(A::AbstractQuasiArray) = LinearIndices(axes(A))



"""
   QSlice(indices)

Represent an axis as a quasi-vector that returns itself.
"""
struct QSlice{T,AX} <: AbstractQuasiVector{T}
    axis::AX
end
QSlice(axis) = QSlice{eltype(axis),typeof(axis)}(axis)
QSlice(S::QSlice) = S
axes(S::QSlice) = (S,)
unsafe_indices(S::QSlice) = (S,)
axes1(S::QSlice) = S
axes(S::QSlice{<:OneTo}) = (S.axis,)
unsafe_indices(S::QSlice{<:OneTo}) = (S.axis,)
axes1(S::QSlice{<:OneTo}) = S.axis

first(S::QSlice) = first(S.axis)
last(S::QSlice) = last(S.axis)
size(S::QSlice) = (length(S.axis),)
length(S::QSlice) = length(S.axis)
unsafe_length(S::QSlice) = unsafe_length(S.axis)
getindex(S::QSlice, i::Real) = (@_inline_meta; @boundscheck checkbounds(S, i); i)
getindex(S::QSlice, i::AbstractVector{<:Real}) = (@_inline_meta; @boundscheck checkbounds(S, i); i)
show(io::IO, r::QSlice) = print(io, "QSlice(", r.axis, ")")
iterate(S::QSlice, s...) = iterate(S.axis, s...)
