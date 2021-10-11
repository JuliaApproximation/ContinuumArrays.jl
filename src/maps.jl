###
# Maps
###

"""
A subtype of `Map` is used as a one-to-one map between two domains
via `view`. The domain of the map `m` is `axes(m,1)` and the range
is `union(m)`.

Maps must also overload `invmap` to give the inverse of the map, which 
is equivalent to `invmap(m)[x] == findfirst(isequal(x), m)`.
"""

abstract type Map{T} <: AbstractQuasiVector{T} end

invmap(M::Map) = error("Overload invmap(::$(typeof(M)))")


Base.in(x, m::Map) = x in union(m)
Base.issubset(d::Map, b) = union(d) ⊆ b
Base.union(d::Map) = axes(invmap(d),1)

for find in (:findfirst, :findlast)
    @eval function $find(f::Base.Fix2{typeof(isequal)}, d::Map)
        f.x in d || return nothing
        $find(isequal(invmap(d)[f.x]), union(d))
    end
end

@eval function findall(f::Base.Fix2{typeof(isequal)}, d::Map)
    f.x in d || return eltype(axes(d,1))[]
    findall(isequal(invmap(d)[f.x]), union(d))
end

function Base.getindex(d::Map, x::Inclusion)
    x == axes(d,1) || throw(BoundsError(d, x))
    d
end

# Affine map represents A*x .+ b
abstract type AbstractAffineQuasiVector{T,AA,X,B} <: Map{T} end

summary(io::IO, a::AbstractAffineQuasiVector) = print(io, "$(a.A) * $(a.x) .+ ($(a.b))")

MemoryLayout(::Type{<:AbstractAffineQuasiVector}) = PolynomialLayout()

struct AffineQuasiVector{T,AA,X,B} <: AbstractAffineQuasiVector{T,AA,X,B}
    A::AA
    x::X
    b::B
end

AffineQuasiVector(A::AA, x::X, b::B) where {AA,X,B} =
    AffineQuasiVector{promote_type(eltype(AA), eltype(X), eltype(B)),AA,X,B}(A,x,b)

AffineQuasiVector(A, x) = AffineQuasiVector(A, x, zero(promote_type(eltype(A),eltype(x))))
AffineQuasiVector(x) = AffineQuasiVector(one(eltype(x)), x)

AffineQuasiVector(A, x::AffineQuasiVector, b) = AffineQuasiVector(A*x.A, x.x, A*x.b .+ b)

axes(A::AbstractAffineQuasiVector) = axes(A.x)

affine_getindex(A, k) = A.A*A.x[k] .+ A.b
Base.unsafe_getindex(A::AbstractAffineQuasiVector, k) = A.A*Base.unsafe_getindex(A.x,k) .+ A.b
getindex(A::AbstractAffineQuasiVector, k::Number) = affine_getindex(A, k)
function getindex(A::AbstractAffineQuasiVector, k::Inclusion)
    @boundscheck A.x[k] # throws bounds error if k ≠ x
    A
end

getindex(A::AbstractAffineQuasiVector, ::Colon) = copy(A)

copy(A::AbstractAffineQuasiVector) = A

inbounds_getindex(A::AbstractAffineQuasiVector{<:Any,<:Any,<:Inclusion}, k::Number) = A.A*k .+ A.b
isempty(A::AbstractAffineQuasiVector) = isempty(A.x)
==(a::AbstractAffineQuasiVector, b::AbstractAffineQuasiVector) = a.A == b.A && a.x == b.x && a.b == b.b
==(a::AbstractAffineQuasiVector, b::Inclusion) =  a == affine(b,b)
==(a::Inclusion, b::AbstractAffineQuasiVector) =  affine(a,a) == b

BroadcastStyle(::Type{<:AbstractAffineQuasiVector}) = LazyQuasiArrayStyle{1}()

for op in(:*, :\, :+, :-)
    @eval broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), a::Number, x::Inclusion) = broadcast($op, a, AffineQuasiVector(x))
end
for op in(:/, :+, :-)
    @eval broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), x::Inclusion, a::Number) = broadcast($op, AffineQuasiVector(x), a)
end

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), a::Number, x::AbstractAffineQuasiVector) = AffineQuasiVector(a, x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(\), a::Number, x::AbstractAffineQuasiVector) = AffineQuasiVector(inv(a), x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(/), x::AbstractAffineQuasiVector, a::Number) = AffineQuasiVector(inv(a), x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(+), a::Number, x::AbstractAffineQuasiVector) = AffineQuasiVector(one(eltype(x)), x, a)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(+), x::AbstractAffineQuasiVector, b::Number) = AffineQuasiVector(one(eltype(x)), x, b)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(-), a::Number, x::AbstractAffineQuasiVector) = AffineQuasiVector(-one(eltype(x)), x, a)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(-), x::AbstractAffineQuasiVector, b::Number) = AffineQuasiVector(one(eltype(x)), x, -b)

Base.@propagate_inbounds _affine_checkindex(inds, r) = isempty(r) | (checkindex(Bool, inds, Base.to_indices(inds, (first(r),))...) & checkindex(Bool, inds,  Base.to_indices(inds, (last(r),))...))
Base.@propagate_inbounds checkindex(::Type{Bool}, inds::Inclusion{<:Any,<:AbstractInterval}, r::Inclusion{<:Any,<:AbstractInterval}) = _affine_checkindex(inds, r)
Base.@propagate_inbounds checkindex(::Type{Bool}, inds::Inclusion{<:Any,<:AbstractInterval}, r::AbstractAffineQuasiVector) = _affine_checkindex(inds, r)

minimum(d::AbstractAffineQuasiVector) = signbit(d.A) ? last(d) : first(d)
maximum(d::AbstractAffineQuasiVector) = signbit(d.A) ? first(d) : last(d)

union(d::AbstractAffineQuasiVector) = Inclusion(minimum(d)..maximum(d))
invmap(d::AbstractAffineQuasiVector) = affine(union(d), axes(d,1))




struct AffineMap{T,D,R} <: AbstractAffineQuasiVector{T,T,D,T}
    domain::D
    range::R
end

AffineMap(domain::AbstractQuasiVector{T}, range::AbstractQuasiVector{V}) where {T,V} =
    AffineMap{promote_type(T,V), typeof(domain),typeof(range)}(domain,range)

measure(x::Inclusion{<:Any,<:AbstractInterval}) = last(x)-first(x)

function getproperty(A::AffineMap, d::Symbol)
    domain, range = getfield(A, :domain), getfield(A, :range)
    d == :x && return domain
    d == :A && return measure(range)/measure(domain)
    d == :b && return (last(domain)*first(range) - first(domain)*last(range))/measure(domain)
    getfield(A, d)
end

function getindex(A::AffineMap{T}, k::Number)  where T
    # ensure we exactly hit range
    k == first(A.domain) && return convert(T, first(A.range))::T
    k == last(A.domain) && return convert(T, last(A.range))::T
    convert(T, affine_getindex(A, k))::T
end


first(A::AffineMap{T}) where T = convert(T, first(A.range))::T
last(A::AffineMap{T}) where T = convert(T, last(A.range))::T

affine(a::AbstractQuasiVector, b::AbstractQuasiVector) = AffineMap(a, b)
affine(a, b::AbstractQuasiVector) = affine(Inclusion(a), b)
affine(a::AbstractQuasiVector, b) = affine(a, Inclusion(b))
affine(a, b) = affine(Inclusion(a), Inclusion(b))


# mapped vectors
const AffineMappedQuasiVector = SubQuasiArray{<:Any, 1, <:Any, <:Tuple{AbstractAffineQuasiVector}}
const AffineMappedQuasiMatrix = SubQuasiArray{<:Any, 2, <:Any, <:Tuple{AbstractAffineQuasiVector,Slice}}

==(a::AffineMappedQuasiVector, b::AffineMappedQuasiVector) = parentindices(a) == parentindices(b) && parent(a) == parent(b)

_sum(V::AffineMappedQuasiVector, ::Colon) = parentindices(V)[1].A \ sum(parent(V))

# pretty print for bases
summary(io::IO, P::AffineMappedQuasiMatrix) = print(io, "$(parent(P)) affine mapped to $(parentindices(P)[1].x.domain)")
summary(io::IO, P::AffineMappedQuasiVector) = print(io, "$(parent(P)) affine mapped to $(parentindices(P)[1].x.domain)")