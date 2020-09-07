abstract type Basis{T} <: LazyQuasiMatrix{T} end
abstract type Weight{T} <: LazyQuasiVector{T} end


const WeightedBasis{T, A<:AbstractQuasiVector, B<:Basis} = BroadcastQuasiMatrix{T,typeof(*),<:Tuple{A,B}}

struct WeightLayout <: AbstractQuasiLazyLayout end
abstract type AbstractBasisLayout <: AbstractQuasiLazyLayout end
struct BasisLayout <: AbstractBasisLayout end
struct SubBasisLayout <: AbstractBasisLayout end
struct MappedBasisLayout <: AbstractBasisLayout end
struct WeightedBasisLayout <: AbstractBasisLayout end
struct SubWeightedBasisLayout <: AbstractBasisLayout end
struct MappedWeightedBasisLayout <: AbstractBasisLayout end

SubBasisLayouts = Union{SubBasisLayout,SubWeightedBasisLayout}
WeightedBasisLayouts = Union{WeightedBasisLayout,SubWeightedBasisLayout,MappedWeightedBasisLayout}
MappedBasisLayouts = Union{MappedBasisLayout,MappedWeightedBasisLayout}

abstract type AbstractAdjointBasisLayout <: AbstractQuasiLazyLayout end
struct AdjointBasisLayout <: AbstractAdjointBasisLayout end
struct AdjointSubBasisLayout <: AbstractAdjointBasisLayout end
struct AdjointMappedBasisLayout <: AbstractAdjointBasisLayout end

MemoryLayout(::Type{<:Basis}) = BasisLayout()
MemoryLayout(::Type{<:Weight}) = WeightLayout()

adjointlayout(::Type, ::AbstractBasisLayout) = AdjointBasisLayout()
adjointlayout(::Type, ::SubBasisLayout) = AdjointSubBasisLayout()
adjointlayout(::Type, ::MappedBasisLayouts) = AdjointMappedBasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::BasisLayout) = WeightedBasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::SubBasisLayout) = WeightedBasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::MappedBasisLayouts) = MappedWeightedBasisLayout()

# A sub of a weight is still a weight
sublayout(::WeightLayout, _) = WeightLayout()

## Weighted basis interface
unweightedbasis(P::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) = last(P.args)
unweightedbasis(V::SubQuasiArray) = view(unweightedbasis(parent(V)), parentindices(V)...)



# Default is lazy
ApplyStyle(::typeof(pinv), ::Type{<:Basis}) = LazyQuasiArrayApplyStyle()
pinv(J::Basis) = apply(pinv,J)


function ==(A::Basis, B::Basis)
    axes(A) == axes(B) && throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))
    false
end


@inline copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(+)}}) = +(broadcast(\,Ref(L.A),arguments(L.B))...)
@inline copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(+)},<:Any,<:AbstractQuasiVector}) = 
    transform_ldiv(L.A, L.B)

@inline function copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(-)}})
    a,b = arguments(L.B)
    (L.A\a)-(L.A\b)
end

@inline copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(-)},<:Any,<:AbstractQuasiVector}) =
    transform_ldiv(L.A, L.B)

function copy(P::Ldiv{<:AbstractBasisLayout,<:AbstractBasisLayout})
    A, B = P.A, P.B
    A == B || throw(ArgumentError("Override copy for $(typeof(A)) \\ $(typeof(B))"))
    SquareEye{eltype(P)}((axes(A,2),))
end
function copy(P::Ldiv{<:SubBasisLayouts,<:SubBasisLayouts})
    A, B = P.A, P.B
    parent(A) == parent(B) ||
        throw(ArgumentError("Override copy for $(typeof(A)) \\ $(typeof(B))"))
    Eye{eltype(P)}((axes(A,2),axes(B,2)))
end

@inline function copy(P::Ldiv{<:MappedBasisLayouts,<:MappedBasisLayouts})
    A, B = P.A, P.B
    demap(A)\demap(B)
end

@inline copy(L::Ldiv{<:AbstractBasisLayout,<:SubBasisLayouts}) = apply(\, L.A, ApplyQuasiArray(L.B))
@inline function copy(L::Ldiv{<:SubBasisLayouts,<:AbstractBasisLayout}) 
    P = parent(L.A)
    kr, jr = parentindices(L.A)
    layout_getindex(apply(\, P, L.B), jr, :) # avoid sparse arrays
end


for Bas1 in (:Basis, :WeightedBasis), Bas2 in (:Basis, :WeightedBasis)
    @eval ==(A::SubQuasiArray{<:Any,2,<:$Bas1}, B::SubQuasiArray{<:Any,2,<:$Bas2}) =
        all(parentindices(A) == parentindices(B)) && parent(A) == parent(B)
end


# expansion
_grid(_, P) = error("Overload Grid")
_grid(::MappedBasisLayout, P) = igetindex.(Ref(parentindices(P)[1]), grid(demap(P)))
_grid(::SubBasisLayout, P) = grid(parent(P))
_grid(::WeightedBasisLayouts, P) = grid(unweightedbasis(P))
grid(P) = _grid(MemoryLayout(typeof(P)), P)

struct TransformFactorization{T,Grid,Plan,IPlan} <: Factorization{T}
    grid::Grid
    plan::Plan
    iplan::IPlan
end

TransformFactorization(grid, plan) = 
    TransformFactorization{promote_type(eltype(grid),eltype(plan)),typeof(grid),typeof(plan),Nothing}(grid, plan, nothing)


TransformFactorization(grid, ::Nothing, iplan) = 
    TransformFactorization{promote_type(eltype(grid),eltype(iplan)),typeof(grid),Nothing,typeof(iplan)}(grid, nothing, iplan)

grid(T::TransformFactorization) = T.grid    

\(a::TransformFactorization{<:Any,<:Any,Nothing}, b::AbstractQuasiVector) = a.iplan \  convert(Array, b[a.grid])
\(a::TransformFactorization, b::AbstractQuasiVector) = a.plan * convert(Array, b[a.grid])

\(a::TransformFactorization{<:Any,<:Any,Nothing}, b::AbstractVector) = a.iplan \  b
\(a::TransformFactorization, b::AbstractVector) = a.plan * b

function _factorize(::AbstractBasisLayout, L)
    p = grid(L)
    TransformFactorization(p, nothing, factorize(L[p,:]))
end

struct ProjectionFactorization{T, FAC<:Factorization{T}, INDS} <: Factorization{T}
    F::FAC
    inds::INDS
end

\(a::ProjectionFactorization, b::AbstractQuasiVector) = (a.F \ b)[a.inds]
\(a::ProjectionFactorization, b::AbstractVector) = (a.F \ b)[a.inds]

_factorize(::SubBasisLayout, L) = ProjectionFactorization(factorize(parent(L)), parentindices(L)[2])
# function _factorize(::MappedBasisLayout, L)
#     kr, jr = parentindices(L)
#     P = parent(L)
#     ProjectionFactorization(factorize(view(P,:,jr)), parentindices(L)[2])
# end

transform_ldiv(A, B, _) = factorize(A) \ B
transform_ldiv(A, B) = transform_ldiv(A, B, axes(A))

copy(L::Ldiv{<:AbstractBasisLayout,<:Any,<:Any,<:AbstractQuasiVector}) =
    transform_ldiv(L.A, L.B)

copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) =
    transform_ldiv(L.A, L.B)


##
# Algebra
##

# struct ExpansionLayout <: MemoryLayout end
# applylayout(::Type{typeof(*)}, ::BasisLayout, _) = ExpansionLayout()

const Expansion{T,Space<:AbstractQuasiMatrix,Coeffs<:AbstractVector} = ApplyQuasiVector{T,typeof(*),<:Tuple{Space,Coeffs}}


basis(v::Expansion) = v.args[1]

for op in (:*, :\)
    @eval function broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), x::Number, f::Expansion)
        S,c = arguments(f)
        S * broadcast($op, x, c)
    end
end
for op in (:*, :/)
    @eval function broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), f::Expansion, x::Number)
        S,c = arguments(f)
        S * broadcast($op, c, x)
    end
end


function _broadcastbasis(::typeof(+), _, _, a, b)
    try
        a ≠ b && error("Overload broadcastbasis(::typeof(+), ::$(typeof(a)), ::$(typeof(b)))")
    catch
        error("Overload broadcastbasis(::typeof(+), ::$(typeof(a)), ::$(typeof(b)))")
    end
    a
end

_broadcastbasis(::typeof(+), ::MappedBasisLayouts, ::MappedBasisLayouts, a, b) = broadcastbasis(+, demap(a), demap(b))[basismap(a), :]

broadcastbasis(::typeof(+), a, b) = _broadcastbasis(+, MemoryLayout(a), MemoryLayout(b), a, b)

broadcastbasis(::typeof(-), a, b) = broadcastbasis(+, a, b)

for op in (:+, :-)
    @eval function broadcasted(::LazyQuasiArrayStyle{1}, ::typeof($op), f::Expansion, g::Expansion)
        S,c = arguments(f)
        T,d = arguments(g)
        ST = broadcastbasis($op, S, T)
        ST * $op((ST \ S) * c , (ST \ T) * d)
    end
end

@eval function ==(f::Expansion, g::Expansion)
    S,c = arguments(f)
    T,d = arguments(g)
    ST = broadcastbasis(+, S, T)
    (ST \ S) * c == (ST \ T) * d
end

## materialize views

# materialize(S::SubQuasiArray{<:Any,2,<:ApplyQuasiArray{<:Any,2,typeof(*),<:Tuple{<:Basis,<:Any}}}) =
#     *(arguments(S)...)



# mass matrix
# y = p(x), dy = p'(x) * dx
# \int_a^b f(y) g(y) dy = \int_{-1}^1 f(p(x))*g(p(x)) * p'(x) dx


_sub_getindex(A, kr, jr) = A[kr, jr]
_sub_getindex(A, ::Slice, ::Slice) = A

@simplify function *(Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}},
             B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    A = Ac'
    PA,PB = parent(A),parent(B)
    kr,jr = parentindices(B)
    _sub_getindex((PA'PB)/kr.A,parentindices(A)[2],jr)
end


# Differentiation of sub-arrays
@simplify function *(A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}})
    P = parent(B)
    (Derivative(axes(P,1))*P)[parentindices(B)...]
end

@simplify function *(A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    P = parent(B)
    kr,jr = parentindices(B)
    (Derivative(axes(P,1))*P*kr.A)[kr,jr]
end

# we represent as a Mul with a banded matrix
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractUnitRange}}) = SubBasisLayout()
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:AbstractUnitRange}}) = MappedBasisLayout()
sublayout(::WeightedBasisLayout, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:AbstractUnitRange}}) = MappedWeightedBasisLayout()
sublayout(::WeightedBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractUnitRange}}) = SubWeightedBasisLayout()

@inline sub_materialize(::AbstractBasisLayout, V::AbstractQuasiArray) = V
@inline sub_materialize(::AbstractBasisLayout, V::AbstractArray) = V

demap(x) = x
demap(x::BroadcastQuasiArray) = BroadcastQuasiArray(x.f, map(demap, arguments(x))...)
demap(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}}) = parent(V)
demap(V::SubQuasiArray{<:Any,1,<:Any,<:Tuple{<:AbstractAffineQuasiVector}}) = parent(V)
function demap(V::SubQuasiArray{<:Any,2}) 
    kr, jr = parentindices(V)
    demap(parent(V)[kr,:])[:,jr]
end

basismap(x::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}}) = parentindices(x)[1]
basismap(x::SubQuasiArray{<:Any,1,<:Any,<:Tuple{<:AbstractAffineQuasiVector}}) = parentindices(x)[1]
basismap(x::BroadcastQuasiArray) = basismap(x.args[1])


##
# SubLayout behaves like ApplyLayout{typeof(*)}

combine_mul_styles(::SubBasisLayouts) = combine_mul_styles(ApplyLayout{typeof(*)}())
_mul_arguments(::SubBasisLayouts, A) = _mul_arguments(ApplyLayout{typeof(*)}(), A)
arguments(::SubBasisLayouts, A) = arguments(ApplyLayout{typeof(*)}(), A)
call(::SubBasisLayouts, ::SubQuasiArray) = *

combine_mul_styles(::AdjointSubBasisLayout) = combine_mul_styles(ApplyLayout{typeof(*)}())
_mul_arguments(::AdjointSubBasisLayout, A) = _mul_arguments(ApplyLayout{typeof(*)}(), A)
arguments(::AdjointSubBasisLayout, A) = arguments(ApplyLayout{typeof(*)}(), A)
call(::AdjointSubBasisLayout, ::SubQuasiArray) = *

copy(M::Mul{AdjointSubBasisLayout,<:SubBasisLayouts}) = apply(*, arguments(M.A)..., arguments(M.B)...)

function arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractUnitRange}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = _BandedMatrix(Ones{Int}(1,length(jr)), axes(A,2), first(jr)-1,1-first(jr))
    A,P
end

####
# sum
####

_sum(V::AbstractQuasiArray, dims) = __sum(MemoryLayout(typeof(V)), V, dims)
_sum(V::AbstractQuasiArray, ::Colon) = __sum(MemoryLayout(typeof(V)), V, :)
sum(V::AbstractQuasiArray; dims=:) = _sum(V, dims)

__sum(L, Vm, _) = error("Override for $L")
function __sum(::SubBasisLayout, Vm, dims) 
    @assert dims == 1
    sum(parent(Vm); dims=dims)[:,parentindices(Vm)[2]]
end
function __sum(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVector, ::Colon)
    a = arguments(LAY, V)
    first(apply(*, sum(a[1]; dims=1), tail(a)...))
end

function __sum(::MappedBasisLayout, V::AbstractQuasiArray, dims)
    kr, jr = parentindices(V)
    @assert kr isa AbstractAffineQuasiVector
    sum(demap(V); dims=dims)/kr.A
end

include("splines.jl")
