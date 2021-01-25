module ContinuumArrays
using IntervalSets, LinearAlgebra, LazyArrays, FillArrays, BandedMatrices, QuasiArrays, InfiniteArrays, StaticArrays, BlockArrays
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
import BlockArrays: block, blockindex
import FillArrays: AbstractFill, getindex_value, SquareEye
import ArrayLayouts: mul
import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle, AbstractQuasiLazyLayout,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle, _factorize
import InfiniteArrays: Infinity, InfAxes

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

checkpoints(d::AbstractInterval) = width(d) .* checkpoints(UnitInterval()) .+ leftendpoint(d)
checkpoints(x::Inclusion) = checkpoints(x.domain)
checkpoints(A::AbstractQuasiMatrix) = checkpoints(axes(A,1))


include("operators.jl")
include("bases/bases.jl")
include("basisconcat.jl")

end
