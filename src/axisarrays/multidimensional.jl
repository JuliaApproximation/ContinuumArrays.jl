@inline function _getindex(l::IndexStyle, A::AbstractAxisArray, I::Union{Real, AbstractArray}...)
    @boundscheck checkbounds(A, I...)
    return _unsafe_getindex(l, _maybe_reshape(l, A, I...), I...)
end


_maybe_reshape(::IndexLinear, A::AbstractAxisArray, I...) = A
_maybe_reshape(::IndexCartesian, A::AbstractAxisVector, I...) = A
@inline _maybe_reshape(::IndexCartesian, A::AbstractAxisArray, I...) = __maybe_reshape(A, index_ndims(I...))
@inline __maybe_reshape(A::AbstractAxisArray{T,N}, ::NTuple{N,Any}) where {T,N} = A
@inline __maybe_reshape(A::AbstractAxisArray, ::NTuple{N,Any}) where {N} = reshape(A, Val(N))

function _unsafe_getindex(::IndexStyle, A::AbstractAxisArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
    # This is specifically not inlined to prevent excessive allocations in type unstable code
    shape = index_shape(I...)
    dest = similar(A, shape)
    map(unsafe_length, axes(dest)) == map(unsafe_length, shape) || throw_checksize_error(dest, shape)
    _unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
    return dest
end

# Always index with the exactly indices provided.
@generated function _unsafe_getindex!(dest::Union{AbstractArray,AbstractAxisArray}, src::AbstractAxisArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
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
