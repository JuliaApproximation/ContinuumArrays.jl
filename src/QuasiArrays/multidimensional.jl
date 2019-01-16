@inline function _getindex(l::IndexStyle, A::AbstractQuasiArray, I::Union{Real, AbstractArray}...)
    @boundscheck checkbounds(A, I...)
    return _unsafe_getindex(l, _maybe_reshape(l, A, I...), I...)
end

# Colons get converted to slices by `uncolon`
@inline to_indices(A, inds, I::Tuple{Colon, Vararg{Any}}) =
    (uncolon(inds, I), to_indices(A, _maybetail(inds), tail(I))...)

@inline index_dimsum(::AbstractQuasiArray{Bool}, I...) = (true, index_dimsum(I...)...)
@inline function index_dimsum(::AbstractQuasiArray{<:Any,N}, I...) where N
    (ntuple(x->true, Val(N))..., index_dimsum(I...)...)
end

slice(d::AbstractVector) = Slice(d)
slice(d) = Inclusion(d)

uncolon(inds::Tuple{},    I::Tuple{Colon, Vararg{Any}}) = slice(OneTo(1))
uncolon(inds::Tuple,      I::Tuple{Colon, Vararg{Any}}) = slice(inds[1])


_maybe_reshape(::IndexLinear, A::AbstractQuasiArray, I...) = A
_maybe_reshape(::IndexCartesian, A::AbstractQuasiVector, I...) = A
@inline _maybe_reshape(::IndexCartesian, A::AbstractQuasiArray, I...) = __maybe_reshape(A, index_ndims(I...))
@inline __maybe_reshape(A::AbstractQuasiArray{T,N}, ::NTuple{N,Any}) where {T,N} = A
@inline __maybe_reshape(A::AbstractQuasiArray, ::NTuple{N,Any}) where {N} = reshape(A, Val(N))

function _unsafe_getindex(::IndexStyle, A::AbstractQuasiArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
    # This is specifically not inlined to prevent excessive allocations in type unstable code
    shape = index_shape(I...)
    dest = similar(A, shape)
    map(unsafe_length, axes(dest)) == map(unsafe_length, shape) || throw_checksize_error(dest, shape)
    _unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
    return dest
end

# Always index with the exactly indices provided.
@generated function _unsafe_getindex!(dest::Union{AbstractArray,AbstractQuasiArray}, src::AbstractQuasiArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
    quote
        @_inline_meta
        D = eachindex(dest)
        Dy = iterate(D)
        @inbounds @nloops $N j d->I[d] begin
            # This condition is never hit, but at the moment
            # the optimizer is not clever enough to split the union without it
            Dy === nothing && return dest
            (idx, state) = Dy
            dest[idx] = @ncall $N getindex src j
            Dy = iterate(D, state)
        end
        return dest
    end
end
