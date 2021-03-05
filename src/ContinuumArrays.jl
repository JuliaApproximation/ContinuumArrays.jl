module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, FillArrays, BandedMatrices, QuasiArrays, Infinities, InfiniteArrays, StaticArrays, BlockArrays
import Base: @_inline_meta, @_propagate_inbounds_meta, axes, getindex, convert, prod, *, /, \, +, -, ==, ^,
                IndexStyle, IndexLinear, ==, OneTo, _maybetail, tail, similar, copyto!, copy, diff,
                first, last, show, isempty, findfirst, findlast, findall, Slice, union, minimum, maximum, sum, _sum,
                getproperty, isone, iszero, zero, abs, <, ≤, >, ≥, string, summary, to_indices
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, most, combine_mul_styles, AbstractArrayApplyStyle,
                        adjointlayout, arguments, _mul_arguments, call, broadcastlayout, layout_getindex, UnknownLayout,
                        sublayout, sub_materialize, ApplyLayout, BroadcastLayout, combine_mul_styles, applylayout,
                        simplifiable, _simplify
import LinearAlgebra: pinv, dot, norm2
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import BlockArrays: block, blockindex, unblock, blockedrange, _BlockedUnitRange, _BlockArray
import FillArrays: AbstractFill, getindex_value, SquareEye
import ArrayLayouts: mul
import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle, AbstractQuasiLazyLayout,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle, _factorize
import InfiniteArrays: Infinity, InfAxes

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, ℵ₁, Inclusion, Basis, WeightedBasis, grid, transform, affine, ..



const QMul2{A,B} = Mul{<:Any, <:Any, <:A,<:B}
const QMul3{A,B,C} = Mul{<:AbstractQuasiArrayApplyStyle, <:Tuple{A,B,C}}

cardinality(::AbstractInterval) = ℵ₁

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


include("maps.jl")

const QInfAxes = Union{Inclusion,AbstractAffineQuasiVector}


sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{Any,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,Any}) = V

# ambiguity error
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{InfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,InfAxes}) = V

#
# BlockQuasiArrays

BlockArrays.blockaxes(::Inclusion) = blockaxes(Base.OneTo(1)) # just use 1 block
function BlockArrays.blockaxes(A::AbstractQuasiArray{T,N}, d) where {T,N}
    @_inline_meta
    d::Integer <= N ? blockaxes(A)[d] : Base.OneTo(1)
end

@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{Block{1}, Vararg{Any}}) =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{BlockRange{1,R}, Vararg{Any}}) where R =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{BlockIndex{1}, Vararg{Any}}) =
    (inds[1][I[1]], to_indices(A, _maybetail(inds), tail(I))...)

@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{AbstractArray{<:BlockIndex{1}}, Vararg{Any}}) =
    (inds[1][I[1]], to_indices(A, _maybetail(inds), tail(I))...)    

checkpoints(d::AbstractInterval{T}) where T = width(d) .* SVector{3,float(T)}(0.823972,0.01,0.3273484) .+ leftendpoint(d)
checkpoints(x::Inclusion) = checkpoints(x.domain)
checkpoints(A::AbstractQuasiMatrix) = checkpoints(axes(A,1))


include("operators.jl")
include("bases/bases.jl")
include("basisconcat.jl")

end
