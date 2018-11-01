

# This file is a part of Julia. License is MIT: https://julialang.org/license

## Basic functions ##

"""
    AbstractQuasiArray{T,N}

Supertype for `N`-dimensional arrays (or array-like types) with elements of type `T`.
[`Array`](@ref) and other types are subtypes of this. See the manual section on the
[`AbstractQuasiArray` interface](@ref man-interface-array).
"""
AbstractQuasiArray

convert(::Type{T}, a::T) where {T<:AbstractQuasiArray} = a
convert(::Type{AbstractQuasiArray{T}}, a::AbstractQuasiArray) where {T} = AbstractQuasiArray{T}(a)
convert(::Type{AbstractQuasiArray{T,N}}, a::AbstractQuasiArray{<:Any,N}) where {T,N} = AbstractQuasiArray{T,N}(a)

"""
    size(A::AbstractQuasiArray, [dim])

Return a tuple containing the dimensions of `A`. Optionally you can specify a
dimension to just get the length of that dimension.

Note that `size` may not be defined for arrays with non-standard indices, in which case [`axes`](@ref)
may be useful. See the manual chapter on [arrays with custom indices](@ref man-custom-indices).

# Examples
```jldoctest
julia> A = fill(1, (2,3,4));

julia> size(A)
(2, 3, 4)

julia> size(A, 2)
3
```
"""
size(t::AbstractQuasiArray{T,N}, d) where {T,N} = d <= N ? size(t)[d] : 1

"""
    axes(A, d)

Return the valid range of indices for array `A` along dimension `d`.

See also [`size`](@ref), and the manual chapter on [arrays with custom indices](@ref man-custom-indices).

# Examples
```jldoctest
julia> A = fill(1, (5,6,7));

julia> axes(A, 2)
Base.OneTo(6)
```
"""
function axes(A::AbstractQuasiArray{T,N}, d) where {T,N}
    @_inline_meta
    d <= N ? axes(A)[d] : OneTo(1)
end


# Performance optimization: get rid of a branch on `d` in `axes(A, d)`
# for d=1. 1d arrays are heavily used, and the first dimension comes up
# in other applications.
axes1(A::AbstractQuasiArray{<:Any,0}) = OneTo(1)
axes1(A::AbstractQuasiArray) = (@_inline_meta; axes(A)[1])

keys(a::AbstractQuasiArray) = CartesianIndices(axes(a))
keys(a::AbstractQuasiVector) = LinearIndices(a)


eltype(::Type{<:AbstractQuasiArray{E}}) where {E} = @isdefined(E) ? E : Any
elsize(A::AbstractQuasiArray) = elsize(typeof(A))

"""
    ndims(A::AbstractQuasiArray) -> Integer

Return the number of dimensions of `A`.

# Examples
```jldoctest
julia> A = fill(1, (3,4,5));

julia> ndims(A)
3
```
"""
ndims(::AbstractQuasiArray{T,N}) where {T,N} = N
ndims(::Type{<:AbstractQuasiArray{T,N}}) where {T,N} = N

"""
    length(collection) -> Integer

Return the number of elements in the collection.

Use [`lastindex`](@ref) to get the last valid index of an indexable collection.

# Examples
```jldoctest
julia> length(1:5)
5

julia> length([1, 2, 3, 4])
4

julia> length([1 2; 3 4])
4
```
"""
length

"""
    length(A::AbstractQuasiArray)

Return the number of elements in the array, defaults to `prod(size(A))`.

# Examples
```jldoctest
julia> length([1, 2, 3, 4])
4

julia> length([1 2; 3 4])
4
```
"""
length(t::AbstractQuasiArray) = (@_inline_meta; prod(size(t)))

# eachindex iterates over all indices. IndexCartesian definitions are later.
eachindex(A::AbstractQuasiVector) = (@_inline_meta(); axes1(A))

"""
    eachindex(A...)

Create an iterable object for visiting each index of an `AbstractQuasiArray` `A` in an efficient
manner. For array types that have opted into fast linear indexing (like `Array`), this is
simply the range `1:length(A)`. For other array types, return a specialized Cartesian
range to efficiently index into the array with indices specified for every dimension. For
other iterables, including strings and dictionaries, return an iterator object
supporting arbitrary index types (e.g. unevenly spaced or non-integer indices).

If you supply more than one `AbstractQuasiArray` argument, `eachindex` will create an
iterable object that is fast for all arguments (a [`UnitRange`](@ref)
if all inputs have fast linear indexing, a [`CartesianIndices`](@ref)
otherwise).
If the arrays have different sizes and/or dimensionalities, `eachindex` will return an
iterable that spans the largest range along each dimension.

# Examples
```jldoctest
julia> A = [1 2; 3 4];

julia> for i in eachindex(A) # linear indexing
           println(i)
       end
1
2
3
4

julia> for i in eachindex(view(A, 1:2, 1:1)) # Cartesian indexing
           println(i)
       end
CartesianIndex(1, 1)
CartesianIndex(2, 1)
```
"""
eachindex(A::AbstractQuasiArray) = (@_inline_meta(); eachindex(IndexStyle(A), A))

function eachindex(A::AbstractQuasiArray, B::AbstractQuasiArray)
    @_inline_meta
    eachindex(IndexStyle(A,B), A, B)
end
function eachindex(A::AbstractQuasiArray, B::AbstractQuasiArray...)
    @_inline_meta
    eachindex(IndexStyle(A,B...), A, B...)
end
eachindex(::IndexLinear, A::AbstractQuasiArray) = (@_inline_meta; OneTo(length(A)))
eachindex(::IndexLinear, A::AbstractQuasiVector) = (@_inline_meta; axes1(A))
function eachindex(::IndexLinear, A::AbstractQuasiArray, B::AbstractQuasiArray...)
    @_inline_meta
    indsA = eachindex(IndexLinear(), A)
    _all_match_first(X->eachindex(IndexLinear(), X), indsA, B...) ||
        throw_eachindex_mismatch(IndexLinear(), A, B...)
    indsA
end


# keys with an IndexStyle
keys(s::IndexStyle, A::AbstractQuasiArray, B::AbstractQuasiArray...) = eachindex(s, A, B...)

"""
    lastindex(collection) -> Real
    lastindex(collection, d) -> Real

Return the last index of `collection`. If `d` is given, return the last index of `collection` along dimension `d`.

The syntaxes `A[end]` and `A[end, end]` lower to `A[lastindex(A)]` and
`A[lastindex(A, 1), lastindex(A, 2)]`, respectively.

# Examples
```jldoctest
julia> lastindex([1,2,4])
3

julia> lastindex(rand(3,4,5), 2)
4
```
"""
lastindex(a::AbstractQuasiArray) = (@_inline_meta; last(eachindex(IndexLinear(), a)))
lastindex(a::AbstractQuasiArray, d) = (@_inline_meta; last(axes(a, d)))

"""
    firstindex(collection) -> Real
    firstindex(collection, d) -> Real

Return the first index of `collection`. If `d` is given, return the first index of `collection` along dimension `d`.

# Examples
```jldoctest
julia> firstindex([1,2,4])
1

julia> firstindex(rand(3,4,5), 2)
1
```
"""
firstindex(a::AbstractQuasiArray) = (@_inline_meta; first(eachindex(IndexLinear(), a)))
firstindex(a::AbstractQuasiArray, d) = (@_inline_meta; first(axes(a, d)))

first(a::AbstractQuasiArray) = a[first(eachindex(a))]
stride(A::AbstractQuasiArray, k::Integer) = strides(A)[k]

function isassigned(a::AbstractQuasiArray, i::Real...)
    try
        a[i...]
        true
    catch e
        if isa(e, BoundsError) || isa(e, UndefRefError)
            return false
        else
            rethrow(e)
        end
    end
end

function checkbounds(::Type{Bool}, A::AbstractQuasiArray, I...)
    @_inline_meta
    checkbounds_indices(Bool, axes(A), I)
end

# Linear indexing is explicitly allowed when there is only one (non-cartesian) index
function checkbounds(::Type{Bool}, A::AbstractQuasiArray, i)
    @_inline_meta
    checkindex(Bool, eachindex(IndexLinear(), A), i)
end
# As a special extension, allow using logical arrays that match the source array exactly
function checkbounds(::Type{Bool}, A::AbstractQuasiArray{<:Any,N}, I::AbstractQuasiArray{Bool,N}) where N
    @_inline_meta
    axes(A) == axes(I)
end

"""
    checkbounds(A, I...)

Throw an error if the specified indices `I` are not in bounds for the given array `A`.
"""
function checkbounds(A::AbstractQuasiArray, I...)
    @_inline_meta
    checkbounds(Bool, A, I...) || throw_boundserror(A, I)
    nothing
end


"""
    checkbounds_indices(Bool, IA, I)

Return `true` if the "requested" indices in the tuple `I` fall within
the bounds of the "permitted" indices specified by the tuple
`IA`. This function recursively consumes elements of these tuples,
usually in a 1-for-1 fashion,

    checkbounds_indices(Bool, (IA1, IA...), (I1, I...)) = checkindex(Bool, IA1, I1) &
                                                          checkbounds_indices(Bool, IA, I)

Note that [`checkindex`](@ref) is being used to perform the actual
bounds-check for a single dimension of the array.

There are two important exceptions to the 1-1 rule: linear indexing and
CartesianIndex{N}, both of which may "consume" more than one element
of `IA`.

See also [`checkbounds`](@ref).
"""
function checkbounds_indices(::Type{Bool}, IA::Tuple, I::Tuple)
    @_inline_meta
    checkindex(Bool, IA[1], I[1]) & checkbounds_indices(Bool, tail(IA), tail(I))
end
function checkbounds_indices(::Type{Bool}, ::Tuple{}, I::Tuple)
    @_inline_meta
    checkindex(Bool, OneTo(1), I[1]) & checkbounds_indices(Bool, (), tail(I))
end
checkbounds_indices(::Type{Bool}, IA::Tuple, ::Tuple{}) = (@_inline_meta; all(x->unsafe_length(x)==1, IA))
checkbounds_indices(::Type{Bool}, ::Tuple{}, ::Tuple{}) = true

throw_boundserror(A, I) = (@_noinline_meta; throw(BoundsError(A, I)))

# check along a single dimension
checkindex(::Type{Bool}, inds, i) = Base.checkindex(Bool, inds, i)


# See also specializations in multidimensional

## Constructors ##

# default arguments to similar()
"""
    similar(array, [element_type=eltype(array)], [dims=size(array)])

Create an uninitialized mutable array with the given element type and size, based upon the
given source array. The second and third arguments are both optional, defaulting to the
given array's `eltype` and `size`. The dimensions may be specified either as a single tuple
argument or as a series of integer arguments.

Custom AbstractQuasiArray subtypes may choose which specific array type is best-suited to return
for the given element type and dimensionality. If they do not specialize this method, the
default is an `Array{element_type}(undef, dims...)`.

For example, `similar(1:10, 1, 4)` returns an uninitialized `Array{Int,2}` since ranges are
neither mutable nor support 2 dimensions:

```julia-repl
julia> similar(1:10, 1, 4)
1×4 Array{Int64,2}:
 4419743872  4374413872  4419743888  0
```

Conversely, `similar(trues(10,10), 2)` returns an uninitialized `BitVector` with two
elements since `BitArray`s are both mutable and can support 1-dimensional arrays:

```julia-repl
julia> similar(trues(10,10), 2)
2-element BitArray{1}:
 false
 false
```

Since `BitArray`s can only store elements of type [`Bool`](@ref), however, if you request a
different element type it will create a regular `Array` instead:

```julia-repl
julia> similar(falses(10), Float64, 2, 4)
2×4 Array{Float64,2}:
 2.18425e-314  2.18425e-314  2.18425e-314  2.18425e-314
 2.18425e-314  2.18425e-314  2.18425e-314  2.18425e-314
```

"""
similar(a::AbstractQuasiArray{T}) where {T}                             = similar(a, T)
similar(a::AbstractQuasiArray, ::Type{T}) where {T}                     = similar(a, T, to_shape(axes(a)))
similar(a::AbstractQuasiArray{T}, dims::Tuple) where {T}                = similar(a, T, to_shape(dims))
similar(a::AbstractQuasiArray{T}, dims::DimOrInd...) where {T}          = similar(a, T, to_shape(dims))
similar(a::AbstractQuasiArray, ::Type{T}, dims::DimOrInd...) where {T}  = similar(a, T, to_shape(dims))
# Similar supports specifying dims as either Integers or AbstractUnitRanges or any mixed combination
# thereof. Ideally, we'd just convert Integers to OneTos and then call a canonical method with the axes,
# but we don't want to require all AbstractQuasiArray subtypes to dispatch on Base.OneTo. So instead we
# define this method to convert supported axes to Ints, with the expectation that an offset array
# package will define a method with dims::Tuple{Union{Integer, UnitRange}, Vararg{Union{Integer, UnitRange}}}
similar(a::AbstractQuasiArray, ::Type{T}, dims::Tuple{Union{Integer, OneTo}, Vararg{Union{Integer, OneTo}}}) where {T} = similar(a, T, to_shape(dims))
# similar creates an Array by default
similar(a::AbstractQuasiArray, ::Type{T}, dims::Dims{N}) where {T,N}    = Array{T,N}(undef, dims)

"""
    similar(storagetype, axes)

Create an uninitialized mutable array analogous to that specified by
`storagetype`, but with `axes` specified by the last
argument. `storagetype` might be a type or a function.

**Examples**:

    similar(Array{Int}, axes(A))

creates an array that "acts like" an `Array{Int}` (and might indeed be
backed by one), but which is indexed identically to `A`. If `A` has
conventional indexing, this will be identical to
`Array{Int}(undef, size(A))`, but if `A` has unconventional indexing then the
indices of the result will match `A`.

    similar(BitArray, (axes(A, 2),))

would create a 1-dimensional logical array whose indices match those
of the columns of `A`.
"""
similar(::Type{T}, dims::DimOrInd...) where {T<:AbstractQuasiArray} = similar(T, dims)
similar(::Type{T}, shape::Tuple{Union{Integer, OneTo}, Vararg{Union{Integer, OneTo}}}) where {T<:AbstractQuasiArray} = similar(T, to_shape(shape))
similar(::Type{T}, dims::Dims) where {T<:AbstractQuasiArray} = T(undef, dims)

"""
    empty(v::AbstractQuasiVector, [eltype])

Create an empty vector similar to `v`, optionally changing the `eltype`.

# Examples

```jldoctest
julia> empty([1.0, 2.0, 3.0])
0-element Array{Float64,1}

julia> empty([1.0, 2.0, 3.0], String)
0-element Array{String,1}
```
"""
empty(a::AbstractQuasiVector{T}, ::Type{U}=T) where {T,U} = Vector{U}()

# like empty, but should return a mutable collection, a Vector by default
emptymutable(a::AbstractQuasiVector{T}, ::Type{U}=T) where {T,U} = Vector{U}()

## from general iterable to any array

function copyto!(dest::AbstractQuasiArray, src)
    destiter = eachindex(dest)
    y = iterate(destiter)
    for x in src
        y === nothing &&
            throw(ArgumentError(string("destination has fewer elements than required")))
        dest[y[1]] = x
        y = iterate(destiter, y[2])
    end
    return dest
end

## copy between abstract arrays - generally more efficient
## since a single index variable can be used.

copyto!(dest::AbstractQuasiArray, src::AbstractQuasiArray) =
    copyto!(IndexStyle(dest), dest, IndexStyle(src), src)

function copyto!(::IndexStyle, dest::AbstractQuasiArray, ::IndexStyle, src::AbstractQuasiArray)
    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
    isempty(srcinds) || (checkbounds(Bool, destinds, first(srcinds)) && checkbounds(Bool, destinds, last(srcinds))) ||
        throw(BoundsError(dest, srcinds))
    @inbounds for i in srcinds
        dest[i] = src[i]
    end
    return dest
end

function copyto!(::IndexStyle, dest::AbstractQuasiArray, ::IndexCartesian, src::AbstractQuasiArray)
    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
    isempty(srcinds) || (checkbounds(Bool, destinds, first(srcinds)) && checkbounds(Bool, destinds, last(srcinds))) ||
        throw(BoundsError(dest, srcinds))
    i = 0
    @inbounds for a in src
        dest[i+=1] = a
    end
    return dest
end

function copy(a::AbstractQuasiArray)
    @_propagate_inbounds_meta
    copymutable(a)
end


"""
    copymutable(a)

Make a mutable copy of an array or iterable `a`.  For `a::Array`,
this is equivalent to `copy(a)`, but for other array types it may
differ depending on the type of `similar(a)`.  For generic iterables
this is equivalent to `collect(a)`.

# Examples
```jldoctest
julia> tup = (1, 2, 3)
(1, 2, 3)

julia> Base.copymutable(tup)
3-element Array{Int64,1}:
 1
 2
 3
```
"""
function copymutable(a::AbstractQuasiArray)
    @_propagate_inbounds_meta
    copyto!(similar(a), a)
end

zero(x::AbstractQuasiArray{T}) where {T} = fill!(similar(x), zero(T))

## iteration support for arrays by iterating over `eachindex` in the array ##
# Allows fast iteration by default for both IndexLinear and IndexCartesian arrays

# While the definitions for IndexLinear are all simple enough to inline on their
# own, IndexCartesian's CartesianIndices is more complicated and requires explicit
# inlining.
function iterate(A::AbstractQuasiArray, state=(eachindex(A),))
    y = iterate(state...)
    y === nothing && return nothing
    A[y[1]], (state[1], tail(y)...)
end

isempty(a::AbstractQuasiArray) = (length(a) == 0)

## Approach:
# We only define one fallback method on getindex for all argument types.
# That dispatches to an (inlined) internal _getindex function, where the goal is
# to transform the indices such that we can call the only getindex method that
# we require the type A{T,N} <: AbstractQuasiArray{T,N} to define; either:
#       getindex(::A, ::Real) # if IndexStyle(A) == IndexLinear() OR
#       getindex(::A{T,N}, ::Vararg{Int, N}) where {T,N} # if IndexCartesian()
# If the subtype hasn't defined the required method, it falls back to the
# _getindex function again where an error is thrown to prevent stack overflows.
"""
    getindex(A, inds...)

Return a subset of array `A` as specified by `inds`, where each `ind` may be an
`Int`, an [`AbstractRange`](@ref), or a [`Vector`](@ref). See the manual section on
[array indexing](@ref man-array-indexing) for details.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Array{Int64,2}:
 1  2
 3  4

julia> getindex(A, 1)
1

julia> getindex(A, [2, 1])
2-element Array{Int64,1}:
 3
 1

julia> getindex(A, 2:4)
3-element Array{Int64,1}:
 3
 2
 4
```
"""
function getindex(A::AbstractQuasiArray, I...)
    @_propagate_inbounds_meta
    error_if_canonical_getindex(IndexStyle(A), A, I...)
    _getindex(IndexStyle(A), A, to_indices(A, I)...)
end
function unsafe_getindex(A::AbstractQuasiArray, I...)
    @_inline_meta
    @inbounds r = getindex(A, I...)
    r
end

error_if_canonical_getindex(::IndexLinear, A::AbstractQuasiArray, ::Real) =
    error("getindex not defined for ", typeof(A))
error_if_canonical_getindex(::IndexCartesian, A::AbstractQuasiArray{T,N}, ::Vararg{Real,N}) where {T,N} =
    error("getindex not defined for ", typeof(A))
error_if_canonical_getindex(::IndexStyle, ::AbstractQuasiArray, ::Any...) = nothing

## Internal definitions
function _getindex(::IndexStyle, A::AbstractQuasiArray, I...)
    materialize(view(A, I...))
end

## IndexLinear Scalar indexing: canonical method is one Int
_getindex(::IndexLinear, A::AbstractQuasiArray, i::Real) = (@_propagate_inbounds_meta; getindex(A, i))
function _getindex(::IndexLinear, A::AbstractQuasiArray, I::Vararg{Real,M}) where M
    @_inline_meta
    @boundscheck checkbounds(A, I...) # generally _to_linear_index requires bounds checking
    @inbounds r = getindex(A, _to_linear_index(A, I...))
    r
end
_to_linear_index(A::AbstractQuasiArray, i::Real) = i
_to_linear_index(A::AbstractQuasiVector, i::Real, I::Real...) = i
_to_linear_index(A::AbstractQuasiArray) = 1
_to_linear_index(A::AbstractQuasiArray, I::Real...) = (@_inline_meta; _sub2ind(A, I...))

## IndexCartesian Scalar indexing: Canonical method is full dimensionality of Reals
function _getindex(::IndexCartesian, A::AbstractQuasiArray, I::Vararg{Real,M}) where M
    @_inline_meta
    @boundscheck checkbounds(A, I...) # generally _to_subscript_indices requires bounds checking
    @inbounds r = getindex(A, _to_subscript_indices(A, I...)...)
    r
end
function _getindex(::IndexCartesian, A::AbstractQuasiArray{T,N}, I::Vararg{Real, N}) where {T,N}
    @_propagate_inbounds_meta
    getindex(A, I...)
end
_to_subscript_indices(A::AbstractQuasiArray, i::Real) = (@_inline_meta; _unsafe_ind2sub(A, i))
_to_subscript_indices(A::AbstractQuasiArray{T,N}) where {T,N} = (@_inline_meta; fill_to_length((), 1, Val(N)))
_to_subscript_indices(A::AbstractQuasiArray{T,0}) where {T} = ()
_to_subscript_indices(A::AbstractQuasiArray{T,0}, i::Real) where {T} = ()
_to_subscript_indices(A::AbstractQuasiArray{T,0}, I::Real...) where {T} = ()
function _to_subscript_indices(A::AbstractQuasiArray{T,N}, I::Real...) where {T,N}
    @_inline_meta
    J, Jrem = IteratorsMD.split(I, Val(N))
    _to_subscript_indices(A, J, Jrem)
end
_to_subscript_indices(A::AbstractQuasiArray, J::Tuple, Jrem::Tuple{}) =
    __to_subscript_indices(A, axes(A), J, Jrem)
function __to_subscript_indices(A::AbstractQuasiArray,
        ::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}}, J::Tuple, Jrem::Tuple{})
    @_inline_meta
    (J..., map(first, tail(_remaining_size(J, axes(A))))...)
end
_to_subscript_indices(A::AbstractQuasiArray{T,N}, I::Vararg{Real,N}) where {T,N} = I

## Setindex! is defined similarly. We first dispatch to an internal _setindex!
# function that allows dispatch on array storage

"""
    setindex!(A, X, inds...)
    A[inds...] = X

Store values from array `X` within some subset of `A` as specified by `inds`.
The syntax `A[inds...] = X` is equivalent to `setindex!(A, X, inds...)`.

# Examples
```jldoctest
julia> A = zeros(2,2);

julia> setindex!(A, [10, 20], [1, 2]);

julia> A[[3, 4]] = [30, 40];

julia> A
2×2 Array{Float64,2}:
 10.0  30.0
 20.0  40.0
```
"""
function setindex!(A::AbstractQuasiArray, v, I...)
    @_propagate_inbounds_meta
    error_if_canonical_setindex(IndexStyle(A), A, I...)
    _setindex!(IndexStyle(A), A, v, to_indices(A, I)...)
end
function unsafe_setindex!(A::AbstractQuasiArray, v, I...)
    @_inline_meta
    @inbounds r = setindex!(A, v, I...)
    r
end

error_if_canonical_setindex(::IndexLinear, A::AbstractQuasiArray, ::Real) =
    error("setindex! not defined for ", typeof(A))
error_if_canonical_setindex(::IndexCartesian, A::AbstractQuasiArray{T,N}, ::Vararg{Real,N}) where {T,N} =
    error("setindex! not defined for ", typeof(A))
error_if_canonical_setindex(::IndexStyle, ::AbstractQuasiArray, ::Any...) = nothing

## Internal definitions
_setindex!(::IndexStyle, A::AbstractQuasiArray, v, I...) =
    error("setindex! for $(typeof(A)) with types $(typeof(I)) is not supported")

## IndexLinear Scalar indexing
_setindex!(::IndexLinear, A::AbstractQuasiArray, v, i::Real) = (@_propagate_inbounds_meta; setindex!(A, v, i))
function _setindex!(::IndexLinear, A::AbstractQuasiArray, v, I::Vararg{Real,M}) where M
    @_inline_meta
    @boundscheck checkbounds(A, I...)
    @inbounds r = setindex!(A, v, _to_linear_index(A, I...))
    r
end

# IndexCartesian Scalar indexing
function _setindex!(::IndexCartesian, A::AbstractQuasiArray{T,N}, v, I::Vararg{Real, N}) where {T,N}
    @_propagate_inbounds_meta
    setindex!(A, v, I...)
end
function _setindex!(::IndexCartesian, A::AbstractQuasiArray, v, I::Vararg{Real,M}) where M
    @_inline_meta
    @boundscheck checkbounds(A, I...)
    @inbounds r = setindex!(A, v, _to_subscript_indices(A, I...)...)
    r
end


parent(a::AbstractQuasiArray) = a


unalias(dest, A::AbstractQuasiArray) = mightalias(dest, A) ? unaliascopy(A) : A
unaliascopy(A::AbstractQuasiArray)::typeof(A) = (@_noinline_meta; _unaliascopy(A, copy(A)))

"""
    Base.mightalias(A::AbstractQuasiArray, B::AbstractQuasiArray)

Perform a conservative test to check if arrays `A` and `B` might share the same memory.

By default, this simply checks if either of the arrays reference the same memory
regions, as identified by their [`Base.dataids`](@ref).
"""
mightalias(A::AbstractQuasiArray, B::AbstractQuasiArray) = !_isdisjoint(dataids(A), dataids(B))

"""
    Base.dataids(A::AbstractQuasiArray)

Return a tuple of `UInt`s that represent the mutable data segments of an array.

Custom arrays that would like to opt-in to aliasing detection of their component
parts can specialize this method to return the concatenation of the `dataids` of
their component parts.  A typical definition for an array that wraps a parent is
`Base.dataids(C::CustomArray) = dataids(C.parent)`.
"""
dataids(A::AbstractQuasiArray) = (UInt(objectid(A)),)


## structured matrix methods ##
replace_in_print_matrix(A::AbstractQuasiMatrix,i::Real,j::Real,s::AbstractString) = s
replace_in_print_matrix(A::AbstractQuasiVector,i::Real,j::Real,s::AbstractString) = s

## Concatenation ##
eltypeof(x::AbstractQuasiArray) = eltype(x)


## Reductions and accumulates ##

function isequal(A::AbstractQuasiArray, B::AbstractQuasiArray)
    if A === B return true end
    if axes(A) != axes(B)
        return false
    end
    for (a, b) in zip(A, B)
        if !isequal(a, b)
            return false
        end
    end
    return true
end

function cmp(A::AbstractQuasiVector, B::AbstractQuasiVector)
    for (a, b) in zip(A, B)
        if !isequal(a, b)
            return isless(a, b) ? -1 : 1
        end
    end
    return cmp(length(A), length(B))
end

isless(A::AbstractQuasiVector, B::AbstractQuasiVector) = cmp(A, B) < 0

function (==)(A::AbstractQuasiArray, B::AbstractQuasiArray)
    if axes(A) != axes(B)
        return false
    end
    anymissing = false
    for (a, b) in zip(A, B)
        eq = (a == b)
        if ismissing(eq)
            anymissing = true
        elseif !eq
            return false
        end
    end
    return anymissing ? missing : true
end

# _sub2ind and _ind2sub
# fallbacks
function _sub2ind(A::AbstractQuasiArray, I...)
    @_inline_meta
    _sub2ind(axes(A), I...)
end

function _ind2sub(A::AbstractQuasiArray, ind)
    @_inline_meta
    _ind2sub(axes(A), ind)
end

# Vectorized forms
function _sub2ind(inds::Indices{1}, I1::AbstractQuasiVector{T}, I::AbstractQuasiVector{T}...) where T<:Real
    throw(ArgumentError("Linear indexing is not defined for one-dimensional arrays"))
end
_sub2ind(inds::Tuple{OneTo}, I1::AbstractQuasiVector{T}, I::AbstractQuasiVector{T}...) where {T<:Real} =
    _sub2ind_vecs(inds, I1, I...)
_sub2ind(inds::Union{DimsInteger,Indices}, I1::AbstractQuasiVector{T}, I::AbstractQuasiVector{T}...) where {T<:Real} =
    _sub2ind_vecs(inds, I1, I...)
function _sub2ind_vecs(inds, I::AbstractQuasiVector...)
    I1 = I[1]
    Iinds = axes1(I1)
    for j = 2:length(I)
        axes1(I[j]) == Iinds || throw(DimensionMismatch("indices of I[1] ($(Iinds)) does not match indices of I[$j] ($(axes1(I[j])))"))
    end
    Iout = similar(I1)
    _sub2ind!(Iout, inds, Iinds, I)
    Iout
end

function _ind2sub(inds::Union{DimsInteger{N},Indices{N}}, ind::AbstractQuasiVector{<:Real}) where N
    M = length(ind)
    t = ntuple(n->similar(ind),Val(N))
    for (i,idx) in pairs(IndexLinear(), ind)
        sub = _ind2sub(inds, idx)
        for j = 1:N
            t[j][i] = sub[j]
        end
    end
    t
end


## map over arrays ##

## transform any set of dimensions
## dims specifies which dimensions will be transformed. for example
## dims==1:2 will call f on all slices A[:,:,...]
"""
    mapslices(f, A; dims)

Transform the given dimensions of array `A` using function `f`. `f` is called on each slice
of `A` of the form `A[...,:,...,:,...]`. `dims` is an integer vector specifying where the
colons go in this expression. The results are concatenated along the remaining dimensions.
For example, if `dims` is `[1,2]` and `A` is 4-dimensional, `f` is called on `A[:,:,i,j]`
for all `i` and `j`.

# Examples
```jldoctest
julia> a = reshape(Vector(1:16),(2,2,2,2))
2×2×2×2 Array{Int64,4}:
[:, :, 1, 1] =
 1  3
 2  4

[:, :, 2, 1] =
 5  7
 6  8

[:, :, 1, 2] =
  9  11
 10  12

[:, :, 2, 2] =
 13  15
 14  16

julia> mapslices(sum, a, dims = [1,2])
1×1×2×2 Array{Int64,4}:
[:, :, 1, 1] =
 10

[:, :, 2, 1] =
 26

[:, :, 1, 2] =
 42

[:, :, 2, 2] =
 58
```
"""
function mapslices(f, A::AbstractQuasiArray; dims)
    if isempty(dims)
        return map(f,A)
    end
    if !isa(dims, AbstractQuasiVector)
        dims = [dims...]
    end

    dimsA = [axes(A)...]
    ndimsA = ndims(A)
    alldims = [1:ndimsA;]

    otherdims = setdiff(alldims, dims)

    idx = Any[first(ind) for ind in axes(A)]
    itershape   = tuple(dimsA[otherdims]...)
    for d in dims
        idx[d] = Slice(axes(A, d))
    end

    # Apply the function to the first slice in order to determine the next steps
    Aslice = A[idx...]
    r1 = f(Aslice)
    # In some cases, we can re-use the first slice for a dramatic performance
    # increase. The slice itself must be mutable and the result cannot contain
    # any mutable containers. The following errs on the side of being overly
    # strict (#18570 & #21123).
    safe_for_reuse = isa(Aslice, StridedArray) &&
                     (isa(r1, Number) || (isa(r1, AbstractQuasiArray) && eltype(r1) <: Number))

    # determine result size and allocate
    Rsize = copy(dimsA)
    # TODO: maybe support removing dimensions
    if !isa(r1, AbstractQuasiArray) || ndims(r1) == 0
        # If the result of f on a single slice is a scalar then we add singleton
        # dimensions. When adding the dimensions, we have to respect the
        # index type of the input array (e.g. in the case of OffsetArrays)
        tmp = similar(Aslice, typeof(r1), reduced_indices(Aslice, 1:ndims(Aslice)))
        tmp[firstindex(tmp)] = r1
        r1 = tmp
    end
    nextra = max(0, length(dims)-ndims(r1))
    if eltype(Rsize) == Int
        Rsize[dims] = [size(r1)..., ntuple(d->1, nextra)...]
    else
        Rsize[dims] = [axes(r1)..., ntuple(d->OneTo(1), nextra)...]
    end
    R = similar(r1, tuple(Rsize...,))

    ridx = Any[map(first, axes(R))...]
    for d in dims
        ridx[d] = axes(R,d)
    end

    concatenate_setindex!(R, r1, ridx...)

    nidx = length(otherdims)
    indices = Iterators.drop(CartesianIndices(itershape), 1) # skip the first element, we already handled it
    inner_mapslices!(safe_for_reuse, indices, nidx, idx, otherdims, ridx, Aslice, A, f, R)
end

concatenate_setindex!(R, X::AbstractQuasiArray, I...) = (R[I...] = X)

## 1 argument

function map!(f::F, dest::AbstractQuasiArray, A::AbstractQuasiArray) where F
    for (i,j) in zip(eachindex(dest),eachindex(A))
        dest[i] = f(A[j])
    end
    return dest
end

# map on collections
map(f, A::AbstractQuasiArray) = collect_similar(A, Generator(f,A))

## 2 argument
function map!(f::F, dest::AbstractQuasiArray, A::AbstractQuasiArray, B::AbstractQuasiArray) where F
    for (i, j, k) in zip(eachindex(dest), eachindex(A), eachindex(B))
        dest[i] = f(A[j], B[k])
    end
    return dest
end


function map_n!(f::F, dest::AbstractQuasiArray, As) where F
    for i = LinearIndices(As[1])
        dest[i] = f(ith_all(i, As)...)
    end
    return dest
end

"""
    map!(function, destination, collection...)

Like [`map`](@ref), but stores the result in `destination` rather than a new
collection. `destination` must be at least as large as the first collection.

# Examples
```jldoctest
julia> x = zeros(3);

julia> map!(x -> x * 2, x, [1, 2, 3]);

julia> x
3-element Array{Float64,1}:
 2.0
 4.0
 6.0
```
"""
map!(f::F, dest::AbstractQuasiArray, As::AbstractQuasiArray...) where {F} = map_n!(f, dest, As)


## hashing AbstractQuasiArray ##

function hash(A::AbstractQuasiArray, h::UInt)
    h = hash(AbstractQuasiArray, h)
    # Axes are themselves AbstractQuasiArrays, so hashing them directly would stack overflow
    # Instead hash the tuple of firsts and lasts along each dimension
    h = hash(map(first, axes(A)), h)
    h = hash(map(last, axes(A)), h)
    isempty(A) && return h

    # Goal: Hash approximately log(N) entries with a higher density of hashed elements
    # weighted towards the end and special consideration for repeated values. Colliding
    # hashes will often subsequently be compared by equality -- and equality between arrays
    # works elementwise forwards and is short-circuiting. This means that a collision
    # between arrays that differ by elements at the beginning is cheaper than one where the
    # difference is towards the end. Furthermore, blindly choosing log(N) entries from a
    # sparse array will likely only choose the same element repeatedly (zero in this case).

    # To achieve this, we work backwards, starting by hashing the last element of the
    # array. After hashing each element, we skip `fibskip` elements, where `fibskip`
    # is pulled from the Fibonacci sequence -- Fibonacci was chosen as a simple
    # ~O(log(N)) algorithm that ensures we don't hit a common divisor of a dimension
    # and only end up hashing one slice of the array (as might happen with powers of
    # two). Finally, we find the next distinct value from the one we just hashed.

    # This is a little tricky since skipping an integer number of values inherently works
    # with linear indices, but `findprev` uses `keys`. Hoist out the conversion "maps":
    ks = keys(A)
    key_to_linear = LinearIndices(ks) # Index into this map to compute the linear index
    linear_to_key = vec(ks)           # And vice-versa

    # Start at the last index
    keyidx = last(ks)
    linidx = key_to_linear[keyidx]
    fibskip = prevfibskip = oneunit(linidx)
    n = 0
    while true
        n += 1
        # Hash the current key-index and its element
        elt = A[keyidx]
        h = hash(keyidx=>elt, h)

        # Skip backwards a Fibonacci number of indices -- this is a linear index operation
        linidx = key_to_linear[keyidx]
        linidx <= fibskip && break
        linidx -= fibskip
        keyidx = linear_to_key[linidx]

        # Only increase the Fibonacci skip once every N iterations. This was chosen
        # to be big enough that all elements of small arrays get hashed while
        # obscenely large arrays are still tractable. With a choice of N=4096, an
        # entirely-distinct 8000-element array will have ~75% of its elements hashed,
        # with every other element hashed in the first half of the array. At the same
        # time, hashing a `typemax(Int64)`-length Float64 range takes about a second.
        if rem(n, 4096) == 0
            fibskip, prevfibskip = fibskip + prevfibskip, fibskip
        end

        # Find a key index with a value distinct from `elt` -- might be `keyidx` itself
        keyidx = findprev(!isequal(elt), A, keyidx)
        keyidx === nothing && break
    end

    return h
end
