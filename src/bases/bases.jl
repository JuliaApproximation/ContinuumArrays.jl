abstract type Basis{T} <: LazyQuasiMatrix{T} end
abstract type Weight{T} <: LazyQuasiVector{T} end


const WeightedBasis{T, A<:AbstractQuasiVector, B<:Basis} = BroadcastQuasiMatrix{T,typeof(*),<:Tuple{A,B}}

struct WeightLayout <: MemoryLayout end
struct BasisLayout <: MemoryLayout end
struct AdjointBasisLayout <: MemoryLayout end

MemoryLayout(::Type{<:Basis}) = BasisLayout()
MemoryLayout(::Type{<:Weight}) = WeightLayout()

adjointlayout(::Type, ::BasisLayout) = AdjointBasisLayout()
transposelayout(::Type{<:Real}, ::BasisLayout) = AdjointBasisLayout()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::BasisLayout) = BasisLayout()

combine_mul_styles(::BasisLayout) = LazyQuasiArrayApplyStyle()
combine_mul_styles(::AdjointBasisLayout) = LazyQuasiArrayApplyStyle()

ApplyStyle(::typeof(pinv), ::Type{<:Basis}) = LazyQuasiArrayApplyStyle()
pinv(J::Basis) = apply(pinv,J)

_multup(a::Tuple) = Mul(a...)
_multup(a) = a


==(A::Basis, B::Basis) = axes(A) ≠ axes(B) ||
    throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))

ApplyStyle(::typeof(\), ::Type{<:Basis}, ::Type{<:AbstractQuasiMatrix}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:Basis}, ::Type{<:AbstractQuasiVector}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:SubQuasiArray{<:Any,2,<:Basis}}, ::Type{<:AbstractQuasiMatrix}) = LdivApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:SubQuasiArray{<:Any,2,<:Basis}}, ::Type{<:AbstractQuasiVector}) = LdivApplyStyle()


for Bas1 in (:Basis, :WeightedBasis), Bas2 in (:Basis, :WeightedBasis)
    @eval begin
        function copy(P::Ldiv{<:Any,<:Any,<:$Bas1,<:$Bas2})
            A, B = P.A, P.B
            A == B || throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
            Eye(size(A,2))
        end
        function copy(P::Ldiv{<:Any,<:Any,<:SubQuasiArray{<:Any,2,<:$Bas1},<:SubQuasiArray{<:Any,2,<:$Bas2}})
            A, B = P.A, P.B
            (parent(A) == parent(B) && parentindices(A) == parentindices(B)) || 
                throw(ArgumentError("Override materialize for $(typeof(A)) \\ $(typeof(B))"))
            Eye(size(A,2))
        end

        function copy(P::Ldiv{<:Any,<:Any,<:SubQuasiArray{<:Any,2,<:$Bas1,<:Tuple{<:AffineQuasiVector,<:Slice}},
                                          <:SubQuasiArray{<:Any,2,<:$Bas2,<:Tuple{<:AffineQuasiVector,<:Slice}}})
            A, B = P.A, P.B
            parent(A)\parent(B)
        end   
        function copy(P::Ldiv{<:Any,<:Any,<:SubQuasiArray{<:Any,2,<:$Bas1,<:Tuple{<:AffineQuasiVector,<:Slice}},
                                          <:SubQuasiArray{<:Any,2,<:$Bas2,<:Tuple{<:AffineQuasiVector,<:Any}}})
            A, B = P.A, P.B
            # use lazy_getindex to avoid sparse arrays
            lazy_getindex(parent(A)\parent(B),:,parentindices(B)[2])
        end        

        function ==(A::SubQuasiArray{<:Any,2,<:$Bas1}, B::SubQuasiArray{<:Any,2,<:$Bas2})
            all(parentindices(A) == parentindices(B)) && parent(A) == parent(B)
        end
    end
end


# expansion
grid(P) = error("Overload Grid")
function transform(L)
    p = grid(L)
    p,L[p,:]
end

function copy(L::Ldiv{BasisLayout,<:Any,<:Any,<:AbstractQuasiVector})
    p,T = transform(L.A)
    T \ L.B[p]
end

function copy(L::Ldiv{BasisLayout,BroadcastLayout{typeof(*)},<:AbstractQuasiMatrix,<:AbstractQuasiVector})
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

function copy(L::Ldiv{BasisLayout,BroadcastLayout{typeof(*)},<:AbstractQuasiMatrix}) 
    args = arguments(L.B)
    # this is a temporary hack
    @assert args[1] isa AbstractQuasiMatrix
    @assert args[2] isa Number
    broadcast(*, L.A \  first(args),  tail(args)...)
end


# we represent as a Mul with a banded matrix
subarraylayout(::BasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractUnitRange}}) = ApplyLayout{typeof(*)}()
subarraylayout(::BasisLayout, ::Type{<:Tuple{<:AffineQuasiVector,<:AbstractUnitRange}}) = BasisLayout()
function arguments(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractUnitRange}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = _BandedMatrix(Ones{Int}(1,length(jr)), axes(A,2), first(jr)-1,1-first(jr))
    A,P
end


include("splines.jl")



