module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, FillArrays, BandedMatrices, QuasiArrays, InfiniteArrays
import Base: @_inline_meta, @_propagate_inbounds_meta, axes, getindex, convert, prod, *, /, \, +, -, ==, ^,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy, diff,
                first, last, show, isempty, findfirst, findlast, findall, Slice, union, minimum, maximum, sum, _sum,
                getproperty, isone, iszero, zero, abs, <, ≤, >, ≥, string, summary
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, most, combine_mul_styles, AbstractArrayApplyStyle,
                        adjointlayout, arguments, _mul_arguments, call, broadcastlayout, layout_getindex, UnknownLayout,
                        sublayout, sub_materialize, ApplyLayout, BroadcastLayout, combine_mul_styles, applylayout,
                        simplifiable, _simplify
import LinearAlgebra: pinv, dot, norm2
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import FillArrays: AbstractFill, getindex_value, SquareEye
import ArrayLayouts: mul
import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle, AbstractQuasiLazyLayout,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle, _factorize
import InfiniteArrays: Infinity

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, ℵ₁, Inclusion, Basis, WeightedBasis, grid, transform, affine

####
# Interval indexing support
####
struct AlephInfinity{N} <: Integer end

isone(::AlephInfinity) = false
iszero(::AlephInfinity) = false

==(::AlephInfinity, ::Int) = false
==(::Int, ::AlephInfinity) = false

*(::AlephInfinity{N}, ::AlephInfinity{N}) where N = AlephInfinity{N}()
*(::AlephInfinity{N}, ::Infinity) where N = AlephInfinity{N}()
*(::Infinity, ::AlephInfinity{N}) where N = AlephInfinity{N}()
function *(a::Integer, b::AlephInfinity)
    a > 0 || throw(ArgumentError("$a is negative"))
    b
end

*(a::AlephInfinity, b::Integer) = b*a


abs(a::AlephInfinity) = a
zero(::AlephInfinity) = 0

for OP in (:<, :≤)
    @eval begin
        $OP(::Real, ::AlephInfinity) = true
        $OP(::AlephInfinity, ::Real) = false
    end
end

for OP in (:>, :≥)
    @eval begin
        $OP(::Real, ::AlephInfinity) = false
        $OP(::AlephInfinity, ::Real) = true
    end
end


const ℵ₁ = AlephInfinity{1}()

string(::AlephInfinity{1}) = "ℵ₁"

show(io::IO, F::AlephInfinity{1}) where N =
    print(io, "ℵ₁")


const QMul2{A,B} = Mul{<:Any, <:Any, <:A,<:B}
const QMul3{A,B,C} = Mul{<:AbstractQuasiArrayApplyStyle, <:Tuple{A,B,C}}

cardinality(::AbstractInterval) = ℵ₁
*(ℵ::AlephInfinity) = ℵ

Inclusion(d::AbstractInterval{T}) where T = Inclusion{float(T)}(d)
first(S::Inclusion{<:Any,<:AbstractInterval}) = leftendpoint(S.domain)
last(S::Inclusion{<:Any,<:AbstractInterval}) = rightendpoint(S.domain)

norm2(x::Inclusion{T,<:AbstractInterval}) where T = sqrt(dot(x,x))

function dot(x::Inclusion{T,<:AbstractInterval}, y::Inclusion{V,<:AbstractInterval}) where {T,V}
    x == y || throw(DimensionMismatch("first quasivector has axis $(x) which does not match the axis of the second, $(y)."))
    TV = promote_type(T,V)
    isempty(x) && return zero(TV)
    a,b = endpoints(x.domain)
    convert(TV, b^3 - a^3)/3
end


function checkindex(::Type{Bool}, inds::Inclusion{<:Any,<:AbstractInterval}, r::Inclusion{<:Any,<:AbstractInterval})
    @_propagate_inbounds_meta
    isempty(r) | (checkindex(Bool, inds, first(r)) & checkindex(Bool, inds, last(r)))
end


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
Base.issubset(d::Map, b::IntervalSets.Domain) = union(d) ⊆ b
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

# Affine map represents A*x .+ b
abstract type AbstractAffineQuasiVector{T,AA,X,B} <: Map{T} end

summary(io::IO, a::AbstractAffineQuasiVector) = print(io, "$(a.A) * $(a.x) .+ ($(a.b))")

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

function checkindex(::Type{Bool}, inds::Inclusion{<:Any,<:AbstractInterval}, r::AbstractAffineQuasiVector)
    @_propagate_inbounds_meta
    isempty(r) | (checkindex(Bool, inds, first(r)) & checkindex(Bool, inds, last(r)))
end

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

measure(x::Inclusion) = last(x)-first(x)

function getproperty(A::AffineMap, d::Symbol)
    domain, range = getfield(A, :domain), getfield(A, :range)
    d == :x && return domain
    d == :A && return measure(range)/measure(domain)
    d == :b && return (last(domain)*first(range) - first(domain)*last(range))/measure(domain)
    getfield(A, d)
end

function getindex(A::AffineMap, k::Number)
    # ensure we exactly hit range
    k == first(A.domain) && return first(A.range)
    k == last(A.domain) && return last(A.range)
    affine_getindex(A, k)
end


first(A::AffineMap) = first(A.range)
last(A::AffineMap) = last(A.range)

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

const QInfAxes = Union{Inclusion,AbstractAffineQuasiVector}


sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{Any,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,Any}) = V


include("operators.jl")
include("bases/bases.jl")

end
