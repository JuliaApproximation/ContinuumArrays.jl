"""
    AbstractConcatBasis

is an abstract type representing a block diagonal basis but
with modified axes.
"""
abstract type AbstractConcatBasis{T} <: Basis{T} end

copy(S::AbstractConcatBasis) = S

function diff(S::AbstractConcatBasis, order...; dims::Integer=1)
    dims == 1 || error("not implemented")
    args = arguments.(Ref(ApplyLayout{typeof(*)}()), diff.(S.args, order...; dims=dims))
    all(length.(args) .== 2) || error("Not implemented")
    concatbasis(S, map(first, args)...) * mortar(Diagonal([map(last, args)...]))
end


"""
    PiecewiseBasis(args...)

is an analogue of `Basis` that takes the union of the first axis,
and the second axis is a blocked concatenatation of args.
If there is overlap, it uses the first in order.
"""
struct PiecewiseBasis{T, Args} <: AbstractConcatBasis{T}
    args::Args
end

PiecewiseBasis{T}(args...) where T = PiecewiseBasis{T,typeof(args)}(args)
PiecewiseBasis(args...) = PiecewiseBasis{mapreduce(eltype,promote_type,args)}(args...)
PiecewiseBasis{T}(args::AbstractVector) where T = PiecewiseBasis{T,typeof(args)}(args)
PiecewiseBasis(args::AbstractVector) = PiecewiseBasis{eltype(eltype(args))}(args)

concatbasis(::PiecewiseBasis, args...) = PiecewiseBasis(args...)


axes(A::PiecewiseBasis) = (union(axes.(A.args,1)...), blockedrange(size.(A.args,2)))

==(A::PiecewiseBasis, B::PiecewiseBasis) = all(A.args .== B.args)


function QuasiArrays._getindex(::Type{IND}, A::PiecewiseBasis{T}, (x,j)::IND) where {IND,T}
    Jj = findblockindex(axes(A,2), j)
    J = Int(block(Jj))
    @boundscheck x in axes(A,1) || throw(BoundsError(A, (x,j)))
    x in axes(A.args[J],1) && return A.args[J][x, blockindex(Jj)]
    zero(T)
end

"""
    f ⊎ g

is the function equal to `f` on `axes(f,1)` and `g` on `axes(g,1)`, where the two
domains are assumed disjoint. Expansions are combined into a single expansion in a
`PiecewiseBasis`.
"""
function ⊎(f, g, h...)
    fs = uplus_flatten((f, g, h...))
    uplus_layout(map(MemoryLayout, fs), fs)
end

function uplus_layout(::Tuple{Vararg{ExpansionLayout}}, fs::Tuple)
    Ps = map(basis, fs)
    uplus_size(map(P -> size(P,2), Ps), Ps, map(coefficients, fs))
end

"""
    uplus_components(f)

returns the pieces `f` is made of, so that `⊎` is associative: `(f ⊎ g) ⊎ h` and
`f ⊎ (g ⊎ h)` both flatten to `⊎(f, g, h)`.
"""
uplus_components(f) = uplus_components_layout(MemoryLayout(f), f)
uplus_components_layout(::ExpansionLayout, f) = uplus_components_basis(basis(f), coefficients(f))

uplus_components_basis(P, c) = (P*c,)
function uplus_components_basis(P::PiecewiseBasis, c)
    ax = axes(P,2)
    map(k -> P.args[k] * c[ax[Block(k)]], ntuple(identity, length(P.args)))
end

uplus_flatten(::Tuple{}) = ()
uplus_flatten(fs::Tuple) = (uplus_components(first(fs))..., uplus_flatten(tail(fs))...)


uplus_basis_size(::NTuple{N,Int}, b) where N = PiecewiseBasis(b...)
uplus_basis(b...) = uplus_basis_size(size.(b, 2), b)

basis_axes(ax::Inclusion{<:Any,<:UnionDomain}, v) = uplus_basis(map(basis, components(ax.domain))...)
coefficient_vcat(P::PiecewiseBasis, cs) = BlockedVector(vcat(cs...), (axes(P,2),))

components_axes(::Inclusion{<:Any,<:UnionDomain}, f) = uplus_components(f)
components_layout(_, f) = components_axes(axes(f,1), f)
DomainSets.components(f::AbstractQuasiVector) = components_layout(MemoryLayout(f), f)

"""
    uplus_axes(ax, Ps, cs)

combines the expansions `Ps .* cs` into a single expansion, dispatching on the tuple
of column axes `ax` so that, for example, infinite bases can interlace the coefficients
instead of concatenating them.
"""
function uplus_size(_, Ps::Tuple, cs::Tuple)
    P = uplus_basis(Ps...)
    P * coefficient_vcat(P, cs)
end

"""
    VcatBasis

is an analogue of `Basis` that vcats the values.
""" 
struct VcatBasis{T, Args} <: AbstractConcatBasis{T}
    args::Args
    function VcatBasis{T, Args}(args::Args) where {T,Args}
        ax = axes(args[1],1)
        for a in Base.tail(args)
            ax == axes(a,1) || throw(ArgumentError("Must be defined on same"))
        end
        new{T,Args}(args)
    end
end

VcatBasis{T}(args...) where T = VcatBasis{T,typeof(args)}(args)
VcatBasis(args::Vararg{Any,N}) where N = VcatBasis{SVector{N,mapreduce(eltype,promote_type,args)}}(args...)

concatbasis(::VcatBasis, args...) = VcatBasis(args...)

axes(A::VcatBasis{<:Any,<:Tuple}) = (axes(A.args[1],1), blockedrange(size.(A.args,2)))

==(A::VcatBasis, B::VcatBasis) = all(A.args .== B.args)

function QuasiArrays._getindex(::Type{IND}, A::VcatBasis{T}, (x,j)::IND) where {IND,T<:SVector{N}} where N
    Jj = findblockindex(axes(A,2), j)
    J = Int(block(Jj))
    T(zeros(J-1)..., A.args[J][x, blockindex(Jj)], zeros(N-J)...) # TODO: type stable via @generated
end

"""
    VcatBasis

is an analogue of `Basis` that hvcats the values, so they are matrix valued.
""" 
struct HvcatBasis{T, Args} <: AbstractConcatBasis{T}
    n::Int
    args::Args
    function HvcatBasis{T, Args}(n::Int, args::Args) where {T,Args}
        ax = axes(args[1],1)
        for a in Base.tail(args)
            ax == axes(a,1) || throw(ArgumentError("Must be defined on same"))
        end
        new{T,Args}(n, args)
    end
end

HvcatBasis{T}(n::Int, args...) where T = HvcatBasis{T,typeof(args)}(n, args)
HvcatBasis(n::Int, args::Vararg{Any,N}) where N = HvcatBasis{Matrix{mapreduce(eltype,promote_type,args)}}(n, args...)

concatbasis(S::HvcatBasis, args...) = HvcatBasis(S.n, args...)

axes(A::HvcatBasis{<:Any,<:Tuple}) = (axes(A.args[1],1), blockedrange(size.(A.args,2)))

==(A::HvcatBasis, B::HvcatBasis) = all(A.args .== B.args)

function QuasiArrays._getindex(::Type{IND}, A::HvcatBasis{T}, (x,j)::IND) where {T,IND}
    Jj = findblockindex(axes(A,2), j)
    J = Int(block(Jj))
    hvcat(A.n, zeros(J-1)..., A.args[J][x, blockindex(Jj)], zeros(length(A.args)-J)...)::T
end


diff(H::ApplyQuasiMatrix{<:Any,typeof(hcat)}, order...; dims::Integer=1) = hcat((diff.(H.args, order...; dims=dims))...)