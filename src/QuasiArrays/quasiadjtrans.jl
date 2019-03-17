# This file is a part of Julia. License is MIT: https://julialang.org/license



### basic definitions (types, aliases, constructors, abstractarray interface, sundry similar)

# note that QuasiAdjoint and QuasiTranspose must be able to wrap not only vectors and matrices
# but also factorizations, rotations, and other linear algebra objects, including
# user-defined such objects. so do not restrict the wrapped type.
struct QuasiAdjoint{T,S} <: AbstractQuasiMatrix{T}
    parent::S
    function QuasiAdjoint{T,S}(A::S) where {T,S}
        checkeltype_adjoint(T, eltype(A))
        new(A)
    end
end
struct QuasiTranspose{T,S} <: AbstractQuasiMatrix{T}
    parent::S
    function QuasiTranspose{T,S}(A::S) where {T,S}
        checkeltype_transpose(T, eltype(A))
        new(A)
    end
end

# basic outer constructors
QuasiAdjoint(A) = QuasiAdjoint{Base.promote_op(adjoint,eltype(A)),typeof(A)}(A)
QuasiTranspose(A) = QuasiTranspose{Base.promote_op(transpose,eltype(A)),typeof(A)}(A)

Base.dataids(A::Union{QuasiAdjoint, QuasiTranspose}) = Base.dataids(A.parent)
Base.unaliascopy(A::Union{QuasiAdjoint,QuasiTranspose}) = typeof(A)(Base.unaliascopy(A.parent))

# wrapping lowercase quasi-constructors
"""
    adjoint(A)

Lazy adjoint (conjugate transposition) (also postfix `'`).
Note that `adjoint` is applied recursively to elements.

This operation is intended for linear algebra usage - for general data manipulation see
[`permutedims`](@ref Base.permutedims).

# Examples
```jldoctest
julia> A = [3+2im 9+2im; 8+7im  4+6im]
2×2 Array{Complex{Int64},2}:
 3+2im  9+2im
 8+7im  4+6im

julia> adjoint(A)
2×2 QuasiAdjoint{Complex{Int64},Array{Complex{Int64},2}}:
 3-2im  8-7im
 9-2im  4-6im
```
"""
adjoint(A::AbstractQuasiVecOrMat) = QuasiAdjoint(A)

"""
    transpose(A)

Lazy transpose. Mutating the returned object should appropriately mutate `A`. Often,
but not always, yields `QuasiTranspose(A)`, where `QuasiTranspose` is a lazy transpose wrapper. Note
that this operation is recursive.

This operation is intended for linear algebra usage - for general data manipulation see
[`permutedims`](@ref Base.permutedims), which is non-recursive.

# Examples
```jldoctest
julia> A = [3+2im 9+2im; 8+7im  4+6im]
2×2 Array{Complex{Int64},2}:
 3+2im  9+2im
 8+7im  4+6im

julia> transpose(A)
2×2 QuasiTranspose{Complex{Int64},Array{Complex{Int64},2}}:
 3+2im  8+7im
 9+2im  4+6im
```
"""
transpose(A::AbstractQuasiVecOrMat) = QuasiTranspose(A)

# unwrapping lowercase quasi-constructors
adjoint(A::QuasiAdjoint) = A.parent
transpose(A::QuasiTranspose) = A.parent
adjoint(A::QuasiTranspose{<:Real}) = A.parent
transpose(A::QuasiAdjoint{<:Real}) = A.parent


# some aliases for internal convenience use
const AdjOrTrans{T,S} = Union{QuasiAdjoint{T,S},QuasiTranspose{T,S}} where {T,S}
const QuasiAdjointAbsVec{T} = QuasiAdjoint{T,<:AbstractQuasiVector}
const QuasiTransposeAbsVec{T} = QuasiTranspose{T,<:AbstractQuasiVector}
const AdjOrTransAbsVec{T} = AdjOrTrans{T,<:AbstractQuasiVector}
const AdjOrTransAbsMat{T} = AdjOrTrans{T,<:AbstractQuasiMatrix}

# for internal use below
wrapperop(A::QuasiAdjoint) = adjoint
wrapperop(A::QuasiTranspose) = transpose

# AbstractQuasiArray interface, basic definitions
length(A::AdjOrTrans) = length(A.parent)
size(v::AdjOrTransAbsVec) = (1, length(v.parent))
size(A::AdjOrTransAbsMat) = reverse(size(A.parent))
axes(v::AdjOrTransAbsVec) = (Base.OneTo(1), axes(v.parent)...)
axes(A::AdjOrTransAbsMat) = reverse(axes(A.parent))
IndexStyle(::Type{<:AdjOrTransAbsVec}) = IndexLinear()
IndexStyle(::Type{<:AdjOrTransAbsMat}) = IndexCartesian()
@propagate_inbounds getindex(v::AdjOrTransAbsVec, i::Real) = wrapperop(v)(v.parent[i-1+first(axes(v.parent)[1])])
@propagate_inbounds getindex(A::AdjOrTransAbsMat, i::Real, j::Real) = wrapperop(A)(A.parent[j, i])
@propagate_inbounds setindex!(v::AdjOrTransAbsVec, x, i::Real) = (setindex!(v.parent, wrapperop(v)(x), i-1+first(axes(v.parent)[1])); v)
@propagate_inbounds setindex!(A::AdjOrTransAbsMat, x, i::Real, j::Real) = (setindex!(A.parent, wrapperop(A)(x), j, i); A)
# AbstractQuasiArray interface, additional definitions to retain wrapper over vectors where appropriate
@propagate_inbounds getindex(v::AdjOrTransAbsVec, ::Colon, is::AbstractArray{<:Real}) = wrapperop(v)(v.parent[is])
@propagate_inbounds getindex(v::AdjOrTransAbsVec, ::Colon, ::Colon) = wrapperop(v)(v.parent[:])

# conversion of underlying storage
convert(::Type{QuasiAdjoint{T,S}}, A::QuasiAdjoint) where {T,S} = QuasiAdjoint{T,S}(convert(S, A.parent))
convert(::Type{QuasiTranspose{T,S}}, A::QuasiTranspose) where {T,S} = QuasiTranspose{T,S}(convert(S, A.parent))

# for vectors, the semantics of the wrapped and unwrapped types differ
# so attempt to maintain both the parent and wrapper type insofar as possible
similar(A::AdjOrTransAbsVec) = wrapperop(A)(similar(A.parent))
similar(A::AdjOrTransAbsVec, ::Type{T}) where {T} = wrapperop(A)(similar(A.parent, Base.promote_op(wrapperop(A), T)))
# for matrices, the semantics of the wrapped and unwrapped types are generally the same
# and as you are allocating with similar anyway, you might as well get something unwrapped
similar(A::AdjOrTrans) = similar(A.parent, eltype(A), axes(A))
similar(A::AdjOrTrans, ::Type{T}) where {T} = similar(A.parent, T, axes(A))
similar(A::AdjOrTrans, ::Type{T}, dims::Dims{N}) where {T,N} = similar(A.parent, T, dims)

# sundry basic definitions
parent(A::AdjOrTrans) = A.parent
vec(v::AdjOrTransAbsVec) = v.parent

cmp(A::AdjOrTransAbsVec, B::AdjOrTransAbsVec) = cmp(parent(A), parent(B))
isless(A::AdjOrTransAbsVec, B::AdjOrTransAbsVec) = isless(parent(A), parent(B))

### concatenation
# preserve QuasiAdjoint/QuasiTranspose wrapper around vectors
# to retain the associated semantics post-concatenation
hcat(avs::Union{Number,QuasiAdjointAbsVec}...) = _adjoint_hcat(avs...)
hcat(tvs::Union{Number,QuasiTransposeAbsVec}...) = _transpose_hcat(tvs...)
_adjoint_hcat(avs::Union{Number,QuasiAdjointAbsVec}...) = adjoint(vcat(map(adjoint, avs)...))
_transpose_hcat(tvs::Union{Number,QuasiTransposeAbsVec}...) = transpose(vcat(map(transpose, tvs)...))
typed_hcat(::Type{T}, avs::Union{Number,QuasiAdjointAbsVec}...) where {T} = adjoint(typed_vcat(T, map(adjoint, avs)...))
typed_hcat(::Type{T}, tvs::Union{Number,QuasiTransposeAbsVec}...) where {T} = transpose(typed_vcat(T, map(transpose, tvs)...))
# otherwise-redundant definitions necessary to prevent hitting the concat methods in sparse/sparsevector.jl
hcat(avs::QuasiAdjoint{<:Any,<:Vector}...) = _adjoint_hcat(avs...)
hcat(tvs::QuasiTranspose{<:Any,<:Vector}...) = _transpose_hcat(tvs...)
hcat(avs::QuasiAdjoint{T,Vector{T}}...) where {T} = _adjoint_hcat(avs...)
hcat(tvs::QuasiTranspose{T,Vector{T}}...) where {T} = _transpose_hcat(tvs...)
# TODO unify and allow mixed combinations


### higher order functions
# preserve QuasiAdjoint/QuasiTranspose wrapper around vectors
# to retain the associated semantics post-map/broadcast
#
# note that the caller's operation f operates in the domain of the wrapped vectors' entries.
# hence the adjoint->f->adjoint shenanigans applied to the parent vectors' entries.
map(f, avs::QuasiAdjointAbsVec...) = adjoint(map((xs...) -> adjoint(f(adjoint.(xs)...)), parent.(avs)...))
map(f, tvs::QuasiTransposeAbsVec...) = transpose(map((xs...) -> transpose(f(transpose.(xs)...)), parent.(tvs)...))


### linear algebra

(-)(A::QuasiAdjoint)   = QuasiAdjoint(  -A.parent)
(-)(A::QuasiTranspose) = QuasiTranspose(-A.parent)

## multiplication *

# QuasiAdjoint/QuasiTranspose-vector * vector
*(u::QuasiAdjointAbsVec, v::AbstractQuasiVector) = dot(u.parent, v)
*(u::QuasiTransposeAbsVec{T}, v::AbstractQuasiVector{T}) where {T<:Real} = dot(u.parent, v)
function *(u::QuasiTransposeAbsVec, v::AbstractQuasiVector)
    @assert !has_offset_axes(u, v)
    @boundscheck length(u) == length(v) || throw(DimensionMismatch())
    return sum(@inbounds(u[k]*v[k]) for k in 1:length(u))
end
# vector * QuasiAdjoint/QuasiTranspose-vector
*(u::AbstractQuasiVector, v::AdjOrTransAbsVec) = broadcast(*, u, v)
# QuasiAdjoint/QuasiTranspose-vector * QuasiAdjoint/QuasiTranspose-vector
# (necessary for disambiguation with fallback methods in linalg/matmul)
*(u::QuasiAdjointAbsVec, v::QuasiAdjointAbsVec) = throw(MethodError(*, (u, v)))
*(u::QuasiTransposeAbsVec, v::QuasiTransposeAbsVec) = throw(MethodError(*, (u, v)))

# AdjOrTransAbsVec{<:Any,<:AdjOrTransAbsVec} is a lazy conj vectors
# We need to expand the combinations to avoid ambiguities
(*)(u::QuasiTransposeAbsVec, v::QuasiAdjointAbsVec{<:Any,<:QuasiTransposeAbsVec}) =
    sum(uu*vv for (uu, vv) in zip(u, v))
(*)(u::QuasiAdjointAbsVec,   v::QuasiAdjointAbsVec{<:Any,<:QuasiTransposeAbsVec}) =
    sum(uu*vv for (uu, vv) in zip(u, v))
(*)(u::QuasiTransposeAbsVec, v::QuasiTransposeAbsVec{<:Any,<:QuasiAdjointAbsVec}) =
    sum(uu*vv for (uu, vv) in zip(u, v))
(*)(u::QuasiAdjointAbsVec,   v::QuasiTransposeAbsVec{<:Any,<:QuasiAdjointAbsVec}) =
    sum(uu*vv for (uu, vv) in zip(u, v))

## pseudoinversion
pinv(v::QuasiAdjointAbsVec, tol::Real = 0) = pinv(v.parent, tol).parent
pinv(v::QuasiTransposeAbsVec, tol::Real = 0) = pinv(conj(v.parent)).parent


## left-division \
\(u::AdjOrTransAbsVec, v::AdjOrTransAbsVec) = pinv(u) * v


## right-division \
/(u::QuasiAdjointAbsVec, A::AbstractQuasiMatrix) = adjoint(adjoint(A) \ u.parent)
/(u::QuasiTransposeAbsVec, A::AbstractQuasiMatrix) = transpose(transpose(A) \ u.parent)
/(u::QuasiAdjointAbsVec, A::QuasiTranspose{<:Any,<:AbstractQuasiMatrix}) = adjoint(conj(A.parent) \ u.parent) # technically should be adjoint(copy(adjoint(copy(A))) \ u.parent)
/(u::QuasiTransposeAbsVec, A::QuasiAdjoint{<:Any,<:AbstractQuasiMatrix}) = transpose(conj(A.parent) \ u.parent) # technically should be transpose(copy(transpose(copy(A))) \ u.parent)


function materialize(M::Mul2{<:Any,<:Any,<:QuasiAdjoint,<:QuasiAdjoint})
    Ac,Bc = M.args
    apply(*,parent(Bc),parent(Ac))'
end

function adjoint(M::Mul)
    Mul(reverse(adjoint.(M.args))...)
end

==(A::QuasiAdjoint, B::QuasiAdjoint) = parent(A) == parent(B)
