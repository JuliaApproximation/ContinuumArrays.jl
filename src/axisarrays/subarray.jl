# This file is a part of Julia. License is MIT: https://julialang.org/license


# L is true if the view itself supports fast linear indexing
struct SubAxisArray{T,N,P,I,L} <: AbstractAxisArray{T,N}
    parent::P
    indices::I
    offset1::Int       # for linear indexing and pointer, only valid when L==true
    stride1::Int       # used only for linear indexing
    function SubAxisArray{T,N,P,I,L}(parent, indices, offset1, stride1) where {T,N,P,I,L}
        @_inline_meta
        check_parent_index_match(parent, indices)
        new(parent, indices, offset1, stride1)
    end
end
# Compute the linear indexability of the indices, and combine it with the linear indexing of the parent
function SubAxisArray(parent::AbstractAxisArray, indices::Tuple)
    @_inline_meta
    SubAxisArray(IndexStyle(viewindexing(indices), IndexStyle(parent)), parent, ensure_indexable(indices), index_dimsum(indices...))
end
function SubAxisArray(::IndexCartesian, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @_inline_meta
    SubAxisArray{eltype(P), N, P, I, false}(parent, indices, 0, 0)
end
function SubAxisArray(::IndexLinear, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @_inline_meta
    # Compute the stride and offset
    stride1 = compute_stride1(parent, indices)
    SubAxisArray{eltype(P), N, P, I, true}(parent, indices, compute_offset1(parent, stride1, indices), stride1)
end

check_parent_index_match(parent::AbstractAxisArray{T,N}, ::NTuple{N, Bool}) where {T,N} = nothing
viewindexing(I::Tuple{AbstractAxisArray, Vararg{Any}}) = IndexCartesian()

# Simple utilities
size(V::SubAxisArray) = (@_inline_meta; map(n->unsafe_length(n), axes(V)))

similar(V::SubAxisArray, T::Type, dims::Dims) = similar(V.parent, T, dims)

sizeof(V::SubAxisArray) = length(V) * sizeof(eltype(V))

parent(V::SubAxisArray) = V.parent
parentindices(V::SubAxisArray) = V.indices
parentindices(a::AbstractAxisArray) = map(OneTo, size(a))

## Aliasing detection
dataids(A::SubAxisArray) = (dataids(A.parent)..., _splatmap(dataids, A.indices)...)
unaliascopy(A::SubAxisArray) = typeof(A)(unaliascopy(A.parent), map(unaliascopy, A.indices), A.offset1, A.stride1)


# Transform indices to be "dense"
_trimmedindex(i::AbstractAxisArray) = oftype(i, reshape(eachindex(IndexLinear(), i), axes(i)))

## SubArray creation
# We always assume that the dimensionality of the parent matches the number of
# indices that end up getting passed to it, so we store the parent as a
# ReshapedArray view if necessary. The trouble is that arrays of `CartesianIndex`
# can make the number of effective indices not equal to length(I).
_maybe_reshape_parent(A::AbstractAxisArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::AbstractAxisArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::AbstractAxisArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
_maybe_reshape_parent(A::AbstractAxisArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))


function view(A::AbstractAxisArray, I::Vararg{Any,N}) where {N}
    @_inline_meta
    J = map(i->unalias(A,i), to_indices(A, I))
    @boundscheck checkbounds(A, J...)
    unsafe_view(_maybe_reshape_parent(A, index_ndims(J...)), J...)
end

function unsafe_view(A::AbstractAxisArray, I::Vararg{ViewIndex,N}) where {N}
    @_inline_meta
    SubAxisArray(A, I)
end
# When we take the view of a view, it's often possible to "reindex" the parent
# view's indices such that we can "pop" the parent view and keep just one layer
# of indirection. But we can't always do this because arrays of `CartesianIndex`
# might span multiple parent indices, making the reindex calculation very hard.
# So we use _maybe_reindex to figure out if there are any arrays of
# `CartesianIndex`, and if so, we punt and keep two layers of indirection.
unsafe_view(V::SubAxisArray, I::Vararg{ViewIndex,N}) where {N} =
    (@_inline_meta; _maybe_reindex(V, I))

## Re-indexing is the heart of a view, transforming A[i, j][x, y] to A[i[x], j[y]]
#
# Recursively look through the heads of the parent- and sub-indices, considering
# the following cases:
# * Parent index is array  -> re-index that with one or more sub-indices (one per dimension)
# * Parent index is Colon  -> just use the sub-index as provided
# * Parent index is scalar -> that dimension was dropped, so skip the sub-index and use the index as is

AbstractZeroDimAxisArray{T} = AbstractAxisArray{T, 0}

# Re-index into parent vectors with one subindex
reindex(V, idxs::Tuple{AbstractAxisVector, Vararg{Any}}, subidxs::Tuple{Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1]], reindex(V, tail(idxs), tail(subidxs))...))

# Parent matrices are re-indexed with two sub-indices
reindex(V, idxs::Tuple{AbstractAxisMatrix, Vararg{Any}}, subidxs::Tuple{Any, Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1], subidxs[2]], reindex(V, tail(idxs), tail(tail(subidxs)))...))

# In general, we index N-dimensional parent arrays with N indices
@generated function reindex(V, idxs::Tuple{AbstractAxisArray{T,N}, Vararg{Any}}, subidxs::Tuple{Vararg{Any}}) where {T,N}
    if length(subidxs.parameters) >= N
        subs = [:(subidxs[$d]) for d in 1:N]
        tail = [:(subidxs[$d]) for d in N+1:length(subidxs.parameters)]
        :(@_propagate_inbounds_meta; (idxs[1][$(subs...)], reindex(V, tail(idxs), ($(tail...),))...))
    else
        :(throw(ArgumentError("cannot re-index $(ndims(V)) dimensional SubAxisArray with fewer than $(ndims(V)) indices\nThis should not occur; please submit a bug report.")))
    end
end

# In general, we simply re-index the parent indices by the provided ones
SlowSubAxisArray{T,N,P,I} = SubAxisArray{T,N,P,I,false}
function getindex(V::SlowSubAxisArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds r = V.parent[reindex(V, V.indices, I)...]
    r
end

FastSubAxisArray{T,N,P,I} = SubAxisArray{T,N,P,I,true}
function getindex(V::FastSubAxisArray, i::Int)
    @_inline_meta
    @boundscheck checkbounds(V, i)
    @inbounds r = V.parent[V.offset1 + V.stride1*i]
    r
end
# We can avoid a multiplication if the first parent index is a Colon or AbstractUnitRange
FastContiguousSubAxisArray{T,N,P,I<:Tuple{Union{Slice, AbstractUnitRange}, Vararg{Any}}} = SubAxisArray{T,N,P,I,true}
function getindex(V::FastContiguousSubAxisArray, i::Int)
    @_inline_meta
    @boundscheck checkbounds(V, i)
    @inbounds r = V.parent[V.offset1 + i]
    r
end

function setindex!(V::SlowSubAxisArray{T,N}, x, I::Vararg{Int,N}) where {T,N}
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds V.parent[reindex(V, V.indices, I)...] = x
    V
end
function setindex!(V::FastSubAxisArray, x, i::Int)
    @_inline_meta
    @boundscheck checkbounds(V, i)
    @inbounds V.parent[V.offset1 + V.stride1*i] = x
    V
end
function setindex!(V::FastContiguousSubAxisArray, x, i::Int)
    @_inline_meta
    @boundscheck checkbounds(V, i)
    @inbounds V.parent[V.offset1 + i] = x
    V
end

IndexStyle(::Type{<:FastSubAxisArray}) = IndexLinear()
IndexStyle(::Type{<:SubAxisArray}) = IndexCartesian()

# Strides are the distance in memory between adjacent elements in a given dimension
# which we determine from the strides of the parent
strides(V::SubAxisArray) = substrides(V.parent, V.indices)
substrides(parent, strds, I::Tuple{Any, Vararg{Any}}) = throw(ArgumentError("strides is invalid for SubAxisArrays with indices of type $(typeof(I[1]))"))

stride(V::SubAxisArray, d::Integer) = d <= ndims(V) ? strides(V)[d] : strides(V)[end] * size(V)[end]

compute_stride1(parent::AbstractAxisArray, I::NTuple{N,Any}) where {N} =
    (@_inline_meta; compute_stride1(1, fill_to_length(axes(parent), OneTo(1), Val(N)), I))

elsize(::Type{<:SubAxisArray{<:Any,<:Any,P}}) where {P} = elsize(P)

iscontiguous(A::SubAxisArray) = iscontiguous(typeof(A))
iscontiguous(::Type{<:SubAxisArray}) = false
iscontiguous(::Type{<:FastContiguousSubAxisArray}) = true

first_index(V::FastSubAxisArray) = V.offset1 + V.stride1 # cached for fast linear SubAxisArrays
function first_index(V::SubAxisArray)
    P, I = parent(V), V.indices
    s1 = compute_stride1(P, I)
    s1 + compute_offset1(P, s1, I)
end

# Computing the first index simply steps through the indices, accumulating the
# sum of index each multiplied by the parent's stride.
# The running sum is `f`; the cumulative stride product is `s`.
# If the parent is a vector, then we offset the parent's own indices with parameters of I
compute_offset1(parent::AbstractAxisVector, stride1::Integer, I::Tuple{AbstractRange}) =
    (@_inline_meta; first(I[1]) - first(axes1(I[1]))*stride1)
# If the result is one-dimensional and it's a Colon, then linear
# indexing uses the indices along the given dimension. Otherwise
# linear indexing always starts with 1.
compute_offset1(parent, stride1::Integer, I::Tuple) =
    (@_inline_meta; compute_offset1(parent, stride1, find_extended_dims(1, I...), find_extended_inds(I...), I))
compute_offset1(parent, stride1::Integer, dims::Tuple{Int}, inds::Tuple{Slice}, I::Tuple) =
    (@_inline_meta; compute_linindex(parent, I) - stride1*first(axes(parent, dims[1])))  # index-preserving case
compute_offset1(parent, stride1::Integer, dims, inds, I::Tuple) =
    (@_inline_meta; compute_linindex(parent, I) - stride1)  # linear indexing starts with 1

function compute_linindex(parent, I::NTuple{N,Any}) where N
    @_inline_meta
    IP = fill_to_length(axes(parent), OneTo(1), Val(N))
    compute_linindex(1, 1, IP, I)
end
function compute_linindex(f, s, IP::Tuple, I::Tuple{ScalarIndex, Vararg{Any}})
    @_inline_meta
    Δi = I[1]-first(IP[1])
    compute_linindex(f + Δi*s, s*unsafe_length(IP[1]), tail(IP), tail(I))
end
function compute_linindex(f, s, IP::Tuple, I::Tuple{Any, Vararg{Any}})
    @_inline_meta
    Δi = first(I[1])-first(IP[1])
    compute_linindex(f + Δi*s, s*unsafe_length(IP[1]), tail(IP), tail(I))
end
compute_linindex(f, s, IP::Tuple, I::Tuple{}) = f

find_extended_dims(dim, ::ScalarIndex, I...) = (@_inline_meta; find_extended_dims(dim + 1, I...))
find_extended_dims(dim, i1, I...) = (@_inline_meta; (dim, find_extended_dims(dim + 1, I...)...))
find_extended_dims(dim) = ()
find_extended_inds(::ScalarIndex, I...) = (@_inline_meta; find_extended_inds(I...))
find_extended_inds(i1, I...) = (@_inline_meta; (i1, find_extended_inds(I...)...))
find_extended_inds() = ()

unsafe_convert(::Type{Ptr{T}}, V::SubAxisArray{T,N,P,<:Tuple{Vararg{RangeIndex}}}) where {T,N,P} =
    unsafe_convert(Ptr{T}, V.parent) + (first_index(V)-1)*sizeof(T)

pointer(V::FastSubAxisArray, i::Int) = pointer(V.parent, V.offset1 + V.stride1*i)
pointer(V::FastContiguousSubAxisArray, i::Int) = pointer(V.parent, V.offset1 + i)
pointer(V::SubAxisArray, i::Int) = _pointer(V, i)
_pointer(V::SubAxisArray{<:Any,1}, i::Int) = pointer(V, (i,))
_pointer(V::SubAxisArray, i::Int) = pointer(V, Base._ind2sub(axes(V), i))


# indices are taken from the range/vector
# Since bounds-checking is performance-critical and uses
# indices, it's worth optimizing these implementations thoroughly
axes(S::SubAxisArray) = (@_inline_meta; _indices_sub(S, S.indices...))
_indices_sub(S::SubAxisArray) = ()
_indices_sub(S::SubAxisArray, ::Real, I...) = (@_inline_meta; _indices_sub(S, I...))
function _indices_sub(S::SubAxisArray, i1::AbstractAxisArray, I...)
    @_inline_meta
    (unsafe_indices(i1)..., _indices_sub(S, I...)...)
end

## Compatibility
# deprecate?
function parentdims(s::SubAxisArray)
    nd = ndims(s)
    dimindex = Vector{Int}(undef, nd)
    sp = strides(s.parent)
    sv = strides(s)
    j = 1
    for i = 1:ndims(s.parent)
        r = s.indices[i]
        if j <= nd && (isa(r,Union{Slice,AbstractRange}) ? sp[i]*step(r) : sp[i]) == sv[j]
            dimindex[j] = i
            j += 1
        end
    end
    dimindex
end
