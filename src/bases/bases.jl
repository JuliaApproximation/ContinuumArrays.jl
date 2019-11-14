abstract type Basis{T} <: LazyQuasiMatrix{T} end
abstract type Weight{T} <: LazyQuasiVector{T} end


const WeightedBasis{T, A<:AbstractQuasiVector, B<:Basis} = BroadcastQuasiMatrix{T,typeof(*),<:Tuple{A,B}}

struct WeightLayout <: MemoryLayout end
abstract type AbstractBasisLayout <: MemoryLayout end
struct BasisLayout <: AbstractBasisLayout end
struct SubBasisLayout <: AbstractBasisLayout end
struct MappedBasisLayout <: AbstractBasisLayout end

abstract type AbstractAdjointBasisLayout <: MemoryLayout end
struct AdjointBasisLayout <: AbstractAdjointBasisLayout end
struct AdjointSubBasisLayout <: AbstractAdjointBasisLayout end
struct AdjointMappedBasisLayout <: AbstractAdjointBasisLayout end

MemoryLayout(::Type{<:Basis}) = BasisLayout()
MemoryLayout(::Type{<:Weight}) = WeightLayout()

adjointlayout(::Type, ::BasisLayout) = AdjointBasisLayout()
adjointlayout(::Type, ::SubBasisLayout) = AdjointSubBasisLayout()
adjointlayout(::Type, ::MappedBasisLayout) = AdjointMappedBasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::BasisLayout) = BasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::SubBasisLayout) = SubBasisLayout()

combine_mul_styles(::AbstractBasisLayout) = LazyQuasiArrayApplyStyle()
combine_mul_styles(::AbstractAdjointBasisLayout) = LazyQuasiArrayApplyStyle()

ApplyStyle(::typeof(pinv), ::Type{<:Basis}) = LazyQuasiArrayApplyStyle()
pinv(J::Basis) = apply(pinv,J)

_multup(a::Tuple) = Mul(a...)
_multup(a) = a


function ==(A::Basis, B::Basis)
    axes(A) == axes(B) && throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))
    false
end


ApplyStyle(::typeof(\), ::Type{<:Basis}, ::Type{<:AbstractQuasiMatrix}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:Basis}, ::Type{<:AbstractQuasiVector}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:SubQuasiArray{<:Any,2,<:Basis}}, ::Type{<:AbstractQuasiMatrix}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:SubQuasiArray{<:Any,2,<:Basis}}, ::Type{<:AbstractQuasiVector}) = LdivApplyStyle()

copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(+)}}) = +(broadcast(\,Ref(L.A),arguments(L.B))...)
copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(+)},<:Any,<:AbstractQuasiVector}) = 
    +(broadcast(\,Ref(L.A),arguments(L.B))...)

function copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(-)}})
    a,b = arguments(L.B)
    (L.A\a)-(L.A\b)
end

function copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(-)},<:Any,<:AbstractQuasiVector})
    a,b = arguments(L.B)
    (L.A\a)-(L.A\b)
end

function copy(P::Ldiv{BasisLayout,BasisLayout})
    A, B = P.A, P.B
    A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end
function copy(P::Ldiv{SubBasisLayout,SubBasisLayout})
    A, B = P.A, P.B
    (parent(A) == parent(B) && parentindices(A) == parentindices(B)) ||
        throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
    Eye(size(A,2))
end

function copy(P::Ldiv{MappedBasisLayout,MappedBasisLayout})
    A, B = P.A, P.B
    demap(A)\demap(B)
end
# function copy(P::Ldiv{MappedBasisLayout,SubBasisLayout})
#     A, B = P.A, P.B
#     # use lazy_getindex to avoid sparse arrays
#     lazy_getindex(parent(A)\parent(B),:,parentindices(B)[2])
# end

for Bas1 in (:Basis, :WeightedBasis), Bas2 in (:Basis, :WeightedBasis)
    @eval ==(A::SubQuasiArray{<:Any,2,<:$Bas1}, B::SubQuasiArray{<:Any,2,<:$Bas2}) =
        all(parentindices(A) == parentindices(B)) && parent(A) == parent(B)
end


# expansion
grid(P) = error("Overload Grid")
function transform(L)
    p = grid(L)
    p,L[p,:]
end

function copy(L::Ldiv{<:AbstractBasisLayout,<:Any,<:Any,<:AbstractQuasiVector})
    p,T = transform(L.A)
    T \ L.B[p]
end

copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) =
    copy(Ldiv{LazyLayout,ApplyLayout{typeof(*)}}(L.A, L.B))

function copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(*)},<:AbstractQuasiMatrix,<:AbstractQuasiVector})
    p,T = transform(L.A)
     T \ L.B[p]
end

## materialize views

# materialize(S::SubQuasiArray{<:Any,2,<:ApplyQuasiArray{<:Any,2,typeof(*),<:Tuple{<:Basis,<:Any}}}) =
#     *(arguments(S)...)


# Differentiation of sub-arrays
function copy(M::QMul2{<:Derivative,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}}})
    A, B = M.args
    P = parent(B)
    (Derivative(axes(P,1))*P)[parentindices(B)...]
end

function copy(M::QMul2{<:Derivative,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AffineQuasiVector,<:Any}}})
    A, B = M.args
    P = parent(B)
    kr,jr = parentindices(B)
    (Derivative(axes(P,1))*P*kr.A)[kr,jr]
end

function copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(*)},<:AbstractQuasiMatrix})
    args = arguments(L.B)
    # this is a temporary hack
    if args isa Tuple{AbstractQuasiMatrix,Number}
        (L.A \  first(args))*last(args)
    elseif args isa Tuple{Number,AbstractQuasiMatrix}
        first(args)*(L.A \ last(args))
    else
        error("Not implemented")
    end
end


# we represent as a Mul with a banded matrix
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractUnitRange}}) = SubBasisLayout()
sublayout(::BasisLayout, ::Type{<:Tuple{<:AffineQuasiVector,<:AbstractUnitRange}}) = MappedBasisLayout()

demap(x) = x
demap(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Any,<:Slice}}) = parent(V)
function demap(V::SubQuasiArray{<:Any,2}) 
    kr, jr = parentindices(V)
    demap(parent(V)[kr,:])[:,jr]
end


##
# SubLayout behaves like ApplyLayout{typeof(*)}

combine_mul_styles(::SubBasisLayout) = combine_mul_styles(ApplyLayout{typeof(*)}())
_arguments(::SubBasisLayout, A) = _arguments(ApplyLayout{typeof(*)}(), A)
call(::SubBasisLayout, ::SubQuasiArray) = *

combine_mul_styles(::AdjointSubBasisLayout) = combine_mul_styles(ApplyLayout{typeof(*)}())
_arguments(::AdjointSubBasisLayout, A) = _arguments(ApplyLayout{typeof(*)}(), A)
arguments(::AdjointSubBasisLayout, A) = arguments(ApplyLayout{typeof(*)}(), A)
call(::AdjointSubBasisLayout, ::SubQuasiArray) = *

function arguments(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractUnitRange}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = _BandedMatrix(Ones{Int}(1,length(jr)), axes(A,2), first(jr)-1,1-first(jr))
    A,P
end

copy(L::Ldiv{BasisLayout,SubBasisLayout}) = apply(\, L.A, ApplyQuasiArray(L.B))



include("splines.jl")
