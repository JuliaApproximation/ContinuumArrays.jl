module ContinuumArrays
using IntervalSets, DomainSets, LinearAlgebra, LazyArrays, FillArrays, BandedMatrices, QuasiArrays, Infinities, InfiniteArrays, StaticArrays, BlockArrays
import Base: @_inline_meta, @_propagate_inbounds_meta, axes, size, getindex, convert, prod, *, /, \, +, -, ==, ^,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy, diff,
                first, last, show, isempty, findfirst, findlast, findall, Slice, union, minimum, maximum, sum, _sum,
                getproperty, isone, iszero, zero, abs, <, ≤, >, ≥, string, summary, to_indices, view, @propagate_inbounds
import Base.Broadcast: materialize, BroadcastStyle, broadcasted, Broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, combine_mul_styles, AbstractArrayApplyStyle,
                        adjointlayout, arguments, _mul_arguments, call, broadcastlayout, layout_getindex, UnknownLayout,
                        sublayout, sub_materialize, ApplyLayout, BroadcastLayout, combine_mul_styles, applylayout,
                        simplifiable, _simplify, AbstractLazyLayout, AbstractPaddedLayout, simplify, Dot
import LinearAlgebra: pinv, inv, dot, norm2, ldiv!, mul!
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import BlockArrays: block, blockindex, unblock, blockedrange, _BlockedUnitRange, _BlockArray
import FillArrays: AbstractFill, getindex_value, SquareEye
import ArrayLayouts: mul, ldiv, ZerosLayout, ScalarLayout, AbstractStridedLayout, check_mul_axes, check_ldiv_axes
import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat, QuasiArrayLayout,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle, AbstractQuasiLazyLayout,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle, _factorize, _cutdim,
                    AbstractQuasiFill, UnionDomain, sum_size, sum_layout, _cumsum, cumsum_layout, applylayout, equals_layout, layout_broadcasted, PolynomialLayout, dot_size,
                    diff_layout, diff_size, AbstractQuasiVecOrMat
import InfiniteArrays: Infinity, InfAxes
import AbstractFFTs: Plan

export Spline, LinearSpline, HeavisideSpline, DiracDelta, Derivative, ℵ₁, Inclusion, Basis, grid, plotgrid, affine, .., transform, expand, plan_transform, basis, coefficients,
        weaklaplacian, laplacian, Laplacian



const QMul2{A,B} = Mul{<:Any, <:Any, <:A,<:B}
const QMul3{A,B,C} = Mul{<:AbstractQuasiArrayApplyStyle, <:Tuple{A,B,C}}

cardinality(::AbstractInterval) = ℵ₁
cardinality(::Union{FullSpace{<:AbstractFloat},EuclideanDomain,DomainSets.RealNumbers,DomainSets.ComplexNumbers}) = ℵ₁
cardinality(::Union{DomainSets.Integers,DomainSets.Rationals,DomainSets.NaturalNumbers}) = ℵ₀

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

sum(x::Inclusion{T,<:AbstractInterval}) where T = convert(T, width(x.domain))


include("maps.jl")

const QInfAxes = Union{Inclusion,AbstractAffineQuasiVector}

# TODO: the following break some tests in QuasiArrays.jl when loaded, when `QInfAxes` are finite dimensional

sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{Any,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,Any}) = V

# ambiguity error
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{InfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,InfAxes}) = V
sub_materialize(::ApplyLayout{typeof(hcat)}, V::AbstractQuasiArray, ::Tuple{QInfAxes,Any}) = V
sub_materialize(::ApplyLayout{typeof(hcat)}, V::AbstractQuasiArray, ::Tuple{QInfAxes,InfAxes}) = V

#
# BlockQuasiArrays

BlockArrays.blockaxes(::Inclusion) = blockaxes(Base.OneTo(1)) # just use 1 block
function BlockArrays.blockaxes(A::AbstractQuasiArray{T,N}, d) where {T,N}
    @_inline_meta
    d::Integer <= N ? blockaxes(A)[d] : Base.OneTo(1)
end

@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{Block{1}, Vararg{Any}}) =
    (unblock(A, inds, I), to_indices(A, _cutdim(inds, I[1]), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{BlockRange{1,R}, Vararg{Any}}) where R =
    (unblock(A, inds, I), to_indices(A, _cutdim(inds, I[1]), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{BlockIndex{1}, Vararg{Any}}) =
    (inds[1][I[1]], to_indices(A, _cutdim(inds, I[1]), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, I::Tuple{BlockRange, Vararg{Any}}) = to_indices(A, axes(A), I)

@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{AbstractArray{<:BlockIndex{1}}, Vararg{Any}}) =
    (inds[1][I[1]], to_indices(A, _cutdim(inds, I[1]), tail(I))...)    

checkpoints(x::Number) = x
checkpoints(d::AbstractInterval{T}) where T = width(d) .* SVector{3,float(T)}(0.823972,0.01,0.3273484) .+ leftendpoint(d)
checkpoints(d::UnionDomain) = mapreduce(checkpoints,union,d.domains)
checkpoints(x::Inclusion) = checkpoints(x.domain)
checkpoints(A::AbstractQuasiMatrix) = checkpoints(axes(A,1))


include("operators.jl")
include("plans.jl")
include("bases/bases.jl")

include("plotting.jl")

###
# sum/dot
###

sum_size(::Tuple{InfiniteCardinal{1}, Vararg{Integer}}, a, dims) = _sum(expand(a), dims)
dot_size(::InfiniteCardinal{1}, a, b) = dot(expand(a), expand(b))
diff_size(::Tuple{InfiniteCardinal{1}, Vararg{Integer}}, a, order...; dims...) = diff(expand(a), order...; dims...)
function copy(d::Dot{<:ExpansionLayout,<:ExpansionLayout,<:AbstractQuasiArray,<:AbstractQuasiArray})
    a,b = d.A,d.B
    P,c = basis(a),coefficients(a)
    Q,d = basis(b),coefficients(b)
    c' * (P'Q) * d
end

end
