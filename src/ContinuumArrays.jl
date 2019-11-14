module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, FillArrays, BandedMatrices, QuasiArrays
import Base: @_inline_meta, @_propagate_inbounds_meta, axes, getindex, convert, prod, *, /, \, +, -, ==,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy,
                first, last, show, isempty, findfirst, findlast, findall, Slice, union, minimum, maximum
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport,
                        adjointlayout, LdivApplyStyle, arguments, _arguments, call, broadcastlayout, lazy_getindex,
                        sublayout, ApplyLayout, BroadcastLayout, combine_mul_styles
import LinearAlgebra: pinv
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import FillArrays: AbstractFill, getindex_value

import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat, quasimulapplystyle,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, fullmaterialize, ℵ₁, Inclusion, Basis, WeightedBasis, grid, transform

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
struct AffineQuasiVector{T,AA,X,B} <: AbstractQuasiVector{T}
    A::AA
    x::X
    b::B
end

AffineQuasiVector(A::AA, x::X, b::B) where {AA,X,B} =
    AffineQuasiVector{promote_type(eltype(AA), eltype(X), eltype(B)),AA,X,B}(A,x,b)

AffineQuasiVector(A, x) = AffineQuasiVector(A, x, zero(promote_type(eltype(A),eltype(x))))
AffineQuasiVector(x) = AffineQuasiVector(one(eltype(x)), x)

AffineQuasiVector(A, x::AffineQuasiVector, b) = AffineQuasiVector(A*x.A, x.x, A*x.b .+ b)

axes(A::AffineQuasiVector) = axes(A.x)
getindex(A::AffineQuasiVector, k::Number) = A.A*A.x[k] .+ A.b
inbounds_getindex(A::AffineQuasiVector{<:Any,<:Any,<:Inclusion}, k::Number) = A.A*k .+ A.b
isempty(A::AffineQuasiVector) = isempty(A.x)
==(a::AffineQuasiVector, b::AffineQuasiVector) = a.A == b.A && a.x == b.x && a.b == b.b

BroadcastStyle(::Type{<:AffineQuasiVector}) = LazyQuasiArrayStyle{1}()

for op in(:*, :\, :+, :-)
    @eval broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), a::Number, x::Inclusion) = broadcast($op, a, AffineQuasiVector(x))
end
for op in(:/, :+, :-)
    @eval broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), x::Inclusion, a::Number) = broadcast($op, AffineQuasiVector(x), a)
end

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), a::Number, x::AffineQuasiVector) = AffineQuasiVector(a, x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(\), a::Number, x::AffineQuasiVector) = AffineQuasiVector(inv(a), x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(/), x::AffineQuasiVector, a::Number) = AffineQuasiVector(inv(a), x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(+), a::Number, x::AffineQuasiVector) = AffineQuasiVector(one(eltype(x)), x, a)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(+), x::AffineQuasiVector, b::Number) = AffineQuasiVector(one(eltype(x)), x, b)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(-), a::Number, x::AffineQuasiVector) = AffineQuasiVector(-one(eltype(x)), x, a)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(-), x::AffineQuasiVector, b::Number) = AffineQuasiVector(one(eltype(x)), x, -b)

function checkindex(::Type{Bool}, inds::Inclusion{<:Any,<:AbstractInterval}, r::AffineQuasiVector{<:Real,<:Real,<:Inclusion{<:Real,<:AbstractInterval}})
    @_propagate_inbounds_meta
    isempty(r) | (checkindex(Bool, inds, first(r)) & checkindex(Bool, inds, last(r)))
end

for find in (:findfirst, :findlast, :findall)
    @eval $find(f::Base.Fix2{typeof(isequal)}, d::AffineQuasiVector) = $find(isequal(d.A \ (f.x .- d.b)), d.x)
end

minimum(d::AffineQuasiVector{<:Real,<:Real,<:Inclusion}) = signbit(d.A) ? last(d) : first(d)
maximum(d::AffineQuasiVector{<:Real,<:Real,<:Inclusion}) = signbit(d.A) ? first(d) : last(d)

union(d::AffineQuasiVector{<:Real,<:Real,<:Inclusion}) = Inclusion(minimum(d)..maximum(d))




include("operators.jl")
include("bases/bases.jl")

end
