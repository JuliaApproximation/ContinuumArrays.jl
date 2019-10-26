module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, FillArrays, BandedMatrices, QuasiArrays
import Base: @_inline_meta, @_propagate_inbounds_meta, axes, getindex, convert, prod, *, /, \, +, -, ==,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy,
                first, last, show, isempty, findfirst, findlast, findall
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, adjointlayout, LdivApplyStyle
import LinearAlgebra: pinv
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import FillArrays: AbstractFill, getindex_value

import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, fullmaterialize, ℵ₁, Inclusion, Basis, WeightedBasis

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


# we represent as a Mul with a banded matrix
function materialize(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractUnitRange}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = _BandedMatrix(Ones{Int}(1,length(jr)), axes(A,2), first(jr)-1,1-first(jr))
    A*P
end

BroadcastStyle(::Type{<:Inclusion{<:Any,<:AbstractInterval}}) = LazyQuasiArrayStyle{1}()
BroadcastStyle(::Type{<:QuasiAdjoint{<:Any,<:Inclusion{<:Any,<:AbstractInterval}}}) = LazyQuasiArrayStyle{2}()
BroadcastStyle(::Type{<:QuasiTranspose{<:Any,<:Inclusion{<:Any,<:AbstractInterval}}}) = LazyQuasiArrayStyle{2}()


###
# Maps
###

# Affine map represents A*x .+ b
struct AffineMap{T,AA,X,B} <: AbstractQuasiVector{T}
    A::AA
    x::X
    b::B
end

AffineMap(A::AA, x::X, b::B) where {AA,X,B} =
    AffineMap{promote_type(eltype(AA), eltype(X), eltype(B)),AA,X,B}(A,x,b)

AffineMap(A, x) = AffineMap(A, x, zero(promote_type(eltype(A),eltype(x))))    
AffineMap(x) = AffineMap(one(eltype(x)), x)

AffineMap(A, x::AffineMap, b) = AffineMap(A*x.A, x.x, A*x.b .+ b)

axes(A::AffineMap) = axes(A.x)
getindex(A::AffineMap, k::Number) = A.A*A.x[k] .+ A.b    
isempty(A::AffineMap) = isempty(A.x)
==(a::AffineMap, b::AffineMap) = a.A == b.A && a.x == b.x && a.b == b.b

BroadcastStyle(::Type{<:AffineMap}) = LazyQuasiArrayStyle{1}()

for op in(:*, :\, :+, :-)
    @eval broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), a::Number, x::Inclusion) = broadcast($op, a, AffineMap(x))
end
for op in(:/, :+, :-)
    @eval broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), x::Inclusion, a::Number) = broadcast(*, AffineMap(x), a)
end

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), a::Number, x::AffineMap) = AffineMap(a, x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(\), a::Number, x::AffineMap) = AffineMap(inv(a), x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(/), x::AffineMap, a::Number) = AffineMap(inv(a), x)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(+), a::Number, x::AffineMap) = AffineMap(one(eltype(x)), x, a)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(+), x::AffineMap, b::Number) = AffineMap(one(eltype(x)), x, b)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(-), a::Number, x::AffineMap) = AffineMap(-one(eltype(x)), x, a)
broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(-), x::AffineMap, b::Number) = AffineMap(one(eltype(x)), x, -b)

function checkindex(::Type{Bool}, inds::Inclusion{<:Any,<:AbstractInterval}, r::AffineMap{<:Real,<:Real,<:Inclusion{<:Real,<:AbstractInterval}})
    @_propagate_inbounds_meta
    isempty(r) | (checkindex(Bool, inds, first(r)) & checkindex(Bool, inds, last(r)))
end


for find in (:findfirst, :findlast, :findall)
    @eval $find(f::Base.Fix2{typeof(isequal)}, d::AffineMap) = $find(isequal(d.A \ (f.x .- d.b)), d.x)
end




include("operators.jl")
include("bases/bases.jl")

end
