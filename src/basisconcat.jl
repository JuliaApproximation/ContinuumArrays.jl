"""
    PiecewiseBlockDiagonal

is an analogue of `BlockDiagonal` that takes the union of the first axis.
If there is overlap, it uses the first in order.
"""
struct PiecewiseBlockDiagonal{T, Args} <: AbstractQuasiMatrix{T}
    args::Args
end

PiecewiseBlockDiagonal{T}(args...) where T = PiecewiseBlockDiagonal{T,typeof(args)}(args)
PiecewiseBlockDiagonal(args...) = PiecewiseBlockDiagonal{mapreduce(eltype,promote_type,args)}(args...)

_blockedrange(a::Tuple) = blockedrange(SVector(a...))
_blockedrange(a::AbstractVector) = blockedrange(a)

axes(A::PiecewiseBlockDiagonal) = (union(axes.(A.args,1)...), _blockedrange(size.(A.args,2)))

function QuasiArrays._getindex(::Type{IND}, A::PiecewiseBlockDiagonal{T}, (x,j)::IND) where {IND,T}
    Jj = findblockindex(axes(A,2), j)
    J = Int(block(Jj))
    @boundscheck x in axes(A,1) || throw(BoundsError(A, (x,j)))
    x in axes(A.args[J],1) && return A.args[J][x, blockindex(Jj)]
    zero(T)
end

"""
    PiecewiseBlockDiagonal

is an analogue of `BlockDiagonal` that vcats the values.
""" 
struct VcatBlockDiagonal{T, Args} <: AbstractQuasiMatrix{T}
    args::Args
    function VcatBlockDiagonal{T, Args}(args::Args) where {T,Args}
        ax = axes(args[1],1)
        for a in Base.tail(args)
            ax == axes(a,1) || throw(ArgumentError("Must be defined on same"))
        end
        new{T,Args}(args)
    end
end

VcatBlockDiagonal{T}(args...) where T = VcatBlockDiagonal{T,typeof(args)}(args)
VcatBlockDiagonal(args::Vararg{Any,N}) where N = VcatBlockDiagonal{SVector{N,mapreduce(eltype,promote_type,args)}}(args...)

axes(A::VcatBlockDiagonal{<:Any,<:Tuple}) = (axes(A.args[1],1), _blockedrange(size.(A.args,2)))

function QuasiArrays._getindex(::Type{IND}, A::VcatBlockDiagonal{T}, (x,j)::IND) where {IND,T<:SVector{N}} where N
    Jj = findblockindex(axes(A,2), j)
    J = Int(block(Jj))
    T(zeros(J-1)..., A.args[J][x, blockindex(Jj)], zeros(N-J)...) # TODO: type stable via @generated
end