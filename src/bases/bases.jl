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
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::AbstractBasisLayout) = WeightedBasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::SubBasisLayout) = WeightedBasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::MappedBasisLayouts) = MappedWeightedBasisLayout()

# A sub of a weight is still a weight
sublayout(::WeightLayout, _) = WeightLayout()
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{Map,AbstractVector}}) = MappedBasisLayout()


## Weighted basis interface
unweightedbasis(P::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) = last(P.args)
unweightedbasis(V::SubQuasiArray) = view(unweightedbasis(parent(V)), parentindices(V)...)
weight(P::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) = first(P.args)
weight(V::SubQuasiArray) = weight(parent(V))[parentindices(V)[1]]



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

@inline function copy(P::Ldiv{<:AbstractBasisLayout,<:AbstractBasisLayout})
    A, B = P.A, P.B
    A == B || throw(ArgumentError("Override copy for $(typeof(A)) \\ $(typeof(B))"))
    SquareEye{eltype(eltype(P))}((axes(A,2),)) # use double eltype for array-valued 
end
@inline function copy(P::Ldiv{<:SubBasisLayouts,<:SubBasisLayouts})
    A, B = P.A, P.B
    parent(A) == parent(B) ||
        throw(ArgumentError("Override copy for $(typeof(A)) \\ $(typeof(B))"))
    Eye{eltype(eltype(P))}((axes(A,2),axes(B,2)))
end

@inline function copy(P::Ldiv{<:MappedBasisLayouts,<:MappedBasisLayouts})
    A, B = P.A, P.B
    demap(A)\demap(B)
end

function copy(P::Ldiv{<:MappedBasisLayouts,<:AbstractLazyLayout})
    A,B = P.A, P.B
    demap(A) \ B[invmap(basismap(A))]
end
copy(P::Ldiv{<:MappedBasisLayouts,ApplyLayout{typeof(*)}}) = copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(P.A,P.B))

@inline copy(L::Ldiv{<:AbstractBasisLayout,<:SubBasisLayouts}) = apply(\, L.A, ApplyQuasiArray(L.B))
@inline function copy(L::Ldiv{<:SubBasisLayouts,<:AbstractBasisLayout}) 
    P = parent(L.A)
    kr, jr = parentindices(L.A)
    layout_getindex(apply(\, P, L.B), jr, :) # avoid sparse arrays
end


for Bas1 in (:Basis, :WeightedBasis), Bas2 in (:Basis, :WeightedBasis)
    @eval ==(A::SubQuasiArray{<:Any,2,<:$Bas1}, B::SubQuasiArray{<:Any,2,<:$Bas2}) =
        parentindices(A) == parentindices(B) && parent(A) == parent(B)
end


# multiplication operators, reexpand in basis A
@inline function _broadcast_mul_ldiv(::Tuple{Any,AbstractBasisLayout}, A, B)
    a,b = arguments(B)
    @assert a isa AbstractQuasiVector # Only works for vec .* mat
    ab = (A * (A \ a)) .* b # broadcasted should be overloaded
    MemoryLayout(ab) isa BroadcastLayout && error("Overload broadcasted(_, ::typeof(*), ::$(typeof(ab.args[1])), ::$(typeof(b)))")
    A \ ab
end

_broadcast_mul_ldiv(_, A, B) = copy(Ldiv{typeof(MemoryLayout(A)),UnknownLayout}(A,B))

copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(*)}}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)
copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)

# ambiguity
copy(L::Ldiv{<:MappedBasisLayouts,BroadcastLayout{typeof(*)}}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)
copy(L::Ldiv{<:MappedBasisLayouts,BroadcastLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)


# expansion
_grid(_, P) = error("Overload Grid")

_grid(::MappedBasisLayout, P) = invmap(parentindices(P)[1])[grid(demap(P))]
_grid(::SubBasisLayout, P) = grid(parent(P))
_grid(::WeightedBasisLayouts, P) = grid(unweightedbasis(P))
grid(P) = _grid(MemoryLayout(P), P)


struct TransformFactorization{T,Grid,Plan,IPlan} <: Factorization{T}
    grid::Grid
    plan::Plan
    iplan::IPlan
end

TransformFactorization{T}(grid, plan) where T = TransformFactorization{T,typeof(grid),typeof(plan),Nothing}(grid, plan, nothing)
TransformFactorization(grid, plan) = TransformFactorization{promote_type(eltype(eltype(grid)),eltype(plan))}(grid, plan)


TransformFactorization{T}(grid, ::Nothing, iplan) where T = TransformFactorization{T,typeof(grid),Nothing,typeof(iplan)}(grid, nothing, iplan)
TransformFactorization(grid, ::Nothing, iplan) = TransformFactorization{promote_type(eltype(eltype(grid)),eltype(iplan))}(grid, nothing, iplan)

grid(T::TransformFactorization) = T.grid    

\(a::TransformFactorization{<:Any,<:Any,Nothing}, b::AbstractQuasiVector{T}) where T = a.iplan \  convert(Array{T}, b[a.grid])
\(a::TransformFactorization, b::AbstractQuasiVector) = a.plan * convert(Array, b[a.grid])
\(a::TransformFactorization{<:Any,<:Any,Nothing}, b::AbstractVector) = a.iplan \  b
\(a::TransformFactorization, b::AbstractVector) = a.plan * b

\(a::TransformFactorization{<:Any,<:Any,Nothing}, b::AbstractQuasiMatrix{T}) where T = a \  convert(Array{T}, b[a.grid,:])
\(a::TransformFactorization, b::AbstractQuasiMatrix) = a \ convert(Array, b[a.grid,:])
function \(a::TransformFactorization, b::AbstractMatrix)
    c = a \ b[:,1]
    ret = Array{eltype(c)}(undef, length(c), size(b,2))
    ret[:,1] = c
    for k = 2:size(ret,2)
        ret[:,k] = a \ b[:,k]
    end
    ret
end

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

struct MappedFactorization{T, FAC<:Factorization{T}, MAP} <: Factorization{T}
    F::FAC
    map::MAP
end

\(a::MappedFactorization, b::AbstractQuasiVector) = a.F \ view(b, a.map)
\(a::MappedFactorization, b::AbstractVector) = a.F \ b

function invmap end

function _factorize(::MappedBasisLayout, L)
    kr, jr = parentindices(L)
    P = parent(L)
    MappedFactorization(factorize(view(P,:,jr)), invmap(parentindices(L)[1]))
end

transform_ldiv(A, B, _) = factorize(A) \ B
transform_ldiv(A, B) = transform_ldiv(A, B, size(A))

copy(L::Ldiv{<:AbstractBasisLayout}) = transform_ldiv(L.A, L.B)
# TODO: redesign to use simplifiable(\, A, B)
copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = transform_ldiv(L.A, L.B)
copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(*)}}) = copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(L.A, L.B))
copy(L::Ldiv{<:AbstractBasisLayout,<:AbstractLazyLayout}) = transform_ldiv(L.A, L.B)

struct WeightedFactorization{T, WW, FAC<:Factorization{T}} <: Factorization{T}
    w::WW
    F::FAC
end

_factorize(::WeightedBasisLayouts, wS) = WeightedFactorization(weight(wS), factorize(unweightedbasis(wS)))


\(F::WeightedFactorization, b::AbstractQuasiVector) = F.F \ (b ./ F.w)

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

function broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), a::Expansion, f::Expansion)
    axes(a,1) == axes(f,1) || throw(DimensionMismatch())
    P,c = arguments(f)
    (a .* P) * c
end


_function_mult_broadcasted(_, _, a, B) = Base.Broadcast.Broadcasted{LazyQuasiArrayStyle{2}}(*, (a, B))
broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::Expansion, B::AbstractQuasiMatrix) = _function_mult_broadcasted(MemoryLayout(a), MemoryLayout(B), a, B)

@eval function ==(f::Expansion, g::Expansion)
    S,c = arguments(f)
    T,d = arguments(g)
    ST = broadcastbasis(+, S, T)
    (ST \ S) * c == (ST \ T) * d
end

function show(io::IO, f::Expansion)
    P,c = arguments(f)
    show(io, P)
    print(io, " * ")
    show(io, c)
end
show(io::IO, ::MIME"text/plain", f::Expansion) = show(io, f)

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

# avoid stack overflow from unmaterialize Derivative() * parent()
_der_sub(DP, inds...) = DP[inds...]
_der_sub(DP::ApplyQuasiMatrix{T,typeof(*),<:Tuple{Derivative,Any}}, kr, jr) where T = ApplyQuasiMatrix{T}(*, DP.args[1], view(DP.args[2], kr, jr))

# need to customise simplifiable so can't use @simplify
simplifiable(::typeof(*), A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}})= simplifiable(*, A, parent(B))
simplifiable(::typeof(*), Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = simplifiable(*, Bc', Ac')
function mul(A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}})
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    P = parent(B)
    _der_sub(Derivative(axes(P,1))*P, parentindices(B)...)
end
mul(Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = mul(Bc', Ac')'

simplifiable(::typeof(*), A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}) = simplifiable(*, A, parent(B))
simplifiable(::typeof(*), Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = simplifiable(*, Bc', Ac')
function mul(A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    P = parent(B)
    kr,jr = parentindices(B)
    (Derivative(axes(P,1))*P*kr.A)[kr,jr]
end
mul(Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = mul(Bc', Ac')'

# we represent as a Mul with a banded matrix
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractVector}}) = SubBasisLayout()
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:AbstractVector}}) = MappedBasisLayout()
sublayout(::WeightedBasisLayouts, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:AbstractVector}}) = MappedWeightedBasisLayout()
sublayout(::WeightedBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractVector}}) = SubWeightedBasisLayout()
sublayout(::MappedWeightedBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractVector}}) = MappedWeightedBasisLayout()

@inline sub_materialize(::AbstractBasisLayout, V::AbstractQuasiArray) = V
@inline sub_materialize(::AbstractBasisLayout, V::AbstractArray) = V

demap(x) = x
demap(x::BroadcastQuasiArray) = BroadcastQuasiArray(x.f, map(demap, arguments(x))...)
demap(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{Any,Slice}}) = parent(V)
demap(V::SubQuasiArray{<:Any,1}) = parent(V)
function demap(V::SubQuasiArray{<:Any,2}) 
    kr, jr = parentindices(V)
    demap(parent(V)[kr,:])[:,jr]
end
function demap(wB::ApplyQuasiArray{<:Any,typeof(*)})
    a = arguments(wB)
    *(demap(first(a)), tail(a)...)
end


basismap(x::SubQuasiArray) = parentindices(x)[1]
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
    P = _BandedMatrix(Ones{Int}((Base.OneTo(1),axes(jr,1))), axes(A,2), first(jr)-1,1-first(jr))
    A,P
end

function arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractVector}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = Eye{Int}((axes(A,2),))[:,jr]
    A,P
end

####
# sum
####


function __sum(::SubBasisLayout, Vm, dims) 
    @assert dims == 1
    sum(parent(Vm); dims=dims)[:,parentindices(Vm)[2]]
end

__sum(::AdjointBasisLayout, Vm::AbstractQuasiMatrix, dims) = permutedims(sum(Vm'; dims=(isone(dims) ? 2 : 1)))


function __sum(::MappedBasisLayouts, V::AbstractQuasiArray, dims)
    kr = basismap(V)
    @assert kr isa AbstractAffineQuasiVector
    sum(demap(V); dims=dims)/kr.A
end

include("splines.jl")
