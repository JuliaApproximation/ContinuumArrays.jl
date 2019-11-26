module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, FillArrays, BandedMatrices, QuasiArrays
import Base: @_inline_meta, @_propagate_inbounds_meta, axes, getindex, convert, prod, *, /, \, +, -, ==,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy, diff,
                first, last, show, isempty, findfirst, findlast, findall, Slice, union, minimum, maximum, sum, _sum,
                getproperty
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport,
                        adjointlayout, LdivApplyStyle, arguments, _arguments, call, broadcastlayout, lazy_getindex,
                        sublayout, sub_materialize, ApplyLayout, BroadcastLayout, combine_mul_styles
import LinearAlgebra: pinv
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import FillArrays: AbstractFill, getindex_value, SquareEye

import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat, quasimulapplystyle,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle, quasildivapplystyle

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, fullmaterialize, ℵ₁, Inclusion, Basis, WeightedBasis, grid, transform, affine

####
# Interval indexing support
####
struct AlephInfinity{N} <: Integer end

==(::AlephInfinity, ::Int) = false
==(::Int, ::AlephInfinity) = false

*(::AlephInfinity{N}, ::AlephInfinity{N}) where N = AlephInfinity{N}()

const ℵ₁ = AlephInfinity{1}()

show(io::IO, F::AlephInfinity{1}) where N =
    print(io, "ℵ₁")


const QMul2{A,B} = Mul{<:AbstractQuasiArrayApplyStyle, <:Tuple{A,B}}
const QMul3{A,B,C} = Mul{<:AbstractQuasiArrayApplyStyle, <:Tuple{A,B,C}}

cardinality(::AbstractInterval) = ℵ₁
*(ℵ::AlephInfinity) = ℵ

Inclusion(d::AbstractInterval{T}) where T = Inclusion{float(T)}(d)
first(S::Inclusion{<:Any,<:AbstractInterval}) = leftendpoint(S.domain)
last(S::Inclusion{<:Any,<:AbstractInterval}) = rightendpoint(S.domain)

for find in (:findfirst, :findlast)
    @eval $find(f::Base.Fix2{typeof(isequal)}, d::Inclusion) = f.x in d.domain ? f.x : nothing
end

function findall(f::Base.Fix2{typeof(isequal)}, d::Inclusion)
    r = findfirst(f,d)
    r === nothing ? eltype(d)[] : [r]
end


function checkindex(::Type{Bool}, inds::Inclusion{<:Any,<:AbstractInterval}, r::Inclusion{<:Any,<:AbstractInterval})
    @_propagate_inbounds_meta
    isempty(r) | (checkindex(Bool, inds, first(r)) & checkindex(Bool, inds, last(r)))
end


BroadcastStyle(::Type{<:Inclusion{<:Any,<:AbstractInterval}}) = LazyQuasiArrayStyle{1}()
BroadcastStyle(::Type{<:QuasiAdjoint{<:Any,<:Inclusion{<:Any,<:AbstractInterval}}}) = LazyQuasiArrayStyle{2}()
BroadcastStyle(::Type{<:QuasiTranspose{<:Any,<:Inclusion{<:Any,<:AbstractInterval}}}) = LazyQuasiArrayStyle{2}()


###
# Maps
###

# Affine map represents A*x .+ b
abstract type AbstractAffineQuasiVector{T,AA,X,B} <: AbstractQuasiVector{T} end

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

affine_igetindex(d, x) = d.A \ (x .- d.b)
igetindex(d::AbstractAffineQuasiVector, x) = affine_igetindex(d, x)

for find in (:findfirst, :findlast, :findall)
    @eval $find(f::Base.Fix2{typeof(isequal)}, d::AbstractAffineQuasiVector) = $find(isequal(igetindex(d, f.x)), d.x)
end

minimum(d::AbstractAffineQuasiVector) = signbit(d.A) ? last(d) : first(d)
maximum(d::AbstractAffineQuasiVector) = signbit(d.A) ? first(d) : last(d)

union(d::AbstractAffineQuasiVector) = Inclusion(minimum(d)..maximum(d))


struct AffineMap{T,D,R} <: AbstractAffineQuasiVector{T,T,D,T}
    domain::D
    range::R
end

AffineMap(domain::AbstractQuasiVector{T}, range::AbstractQuasiVector{V}) where {T,V} = 
    AffineMap{promote_type(T,V), typeof(domain),typeof(range)}(domain,range)

measure(x::Inclusion) = last(x)-first(x)

function getproperty(A::AffineMap, d::Symbol)
    d == :x && return A.domain
    d == :A && return measure(A.range)/measure(A.domain)
    d == :b && return (last(A.domain)*first(A.range) - first(A.domain)*last(A.range))/measure(A.domain)
    getfield(A, d)
end

function getindex(A::AffineMap, k::Number)
    # ensure we exactly hit range
    k == first(A.domain) && return first(A.range)
    k == last(A.domain) && return last(A.range)
    affine_getindex(A, k)
end

function igetindex(A::AffineMap, k::Number)
    # ensure we exactly hit range
    k == first(A.range) && return first(A.domain)
    k == last(A.range) && return last(A.domain)
    affine_igetindex(A, k)
end

first(A::AffineMap) = first(A.range)
last(A::AffineMap) = last(A.range)

affine(a::AbstractQuasiVector, b::AbstractQuasiVector) = AffineMap(a, b)
affine(a, b::AbstractQuasiVector) = affine(Inclusion(a), b)
affine(a::AbstractQuasiVector, b) = affine(a, Inclusion(b))
affine(a, b) = affine(Inclusion(a), Inclusion(b))



const QInfAxes = Union{Inclusion,AbstractAffineQuasiVector}

sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{<:Any,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,Any}) = V


include("operators.jl")
include("bases/bases.jl")

end
