using ContinuumArrays, LazyArrays, IntervalSets
import ContinuumArrays: AbstractAxisMatrix, ℵ₀
import Base: axes, getindex, convert

struct Spline{order,T} <: AbstractAxisMatrix{T}
    points::Vector{T}
end

const LinearSpline{T} = Spline{1,T}
const HeavisideSpline{T} = Spline{0,T}

Spline{o}(pts::AbstractVector{T}) where {o,T} = Spline{o,float(T)}(pts)

axes(B::Spline{o}) where o = (first(B.points)..last(B.points), Base.OneTo(length(B.points)-o-1))

function getindex(B::LinearSpline{T}, x::Real, k::Int) where T
    x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2)|| throw(BoundsError())

    p = B.points
    n = length(p)

    k > 1 && x ≤ p[k-1] && return zero(T)
    k < n && x ≥ p[k+1] && return zero(T)
    x == p[k] && return one(T)
    x < p[k] && return (x-p[k-1])/(p[k]-p[k-1])
    return (x-p[k+1])/(p[k]-p[k+1]) # x ≥ p[k]
end

function getindex(B::HeavisideSpline{T}, x::Real, k::Int) where T
    x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2)|| throw(BoundsError())

    p = B.points
    n = length(p)

    x < p[k] && return zero(T)
    k < n && x > p[k+1] && return zero(T)
    return one(T)
end

# getindex(B::LinearSpline, ::Colon, k::Int) = Mul(B, [Zeros{Int}(k-1); 1; Zeros{Int}(size(B,2)-k)])

# function convert(::Type{SymTridiagonal}, AB::Mul{<:Any,<:Any,<:Any,<:ContinuumArrays.Adjoint{<:Any,<:LinearSpline},<:LinearSpline})
#     Ac,B = AB.A, AB.B
#     A = parent(Ac)
#     @assert A.points == B.points
#     x = A.points
#     SymTridiagonal(x, x/2) # TODO fix
# end
#
# materialize(M::Mul{<:Any,<:Any,<:Any,<:ContinuumArrays.Adjoint{<:Any,<:LinearSpline},<:LinearSpline}) =
#     convert(SymTridiagonal, M)

## tests

B = HeavisideSpline([1,2,3])
@test size(B) == (ℵ₀, 2)

@test_throws BoundsError B[0.1, 1]
@test B[1.1,1] === 1.0
@test B[2.1,1] === 0.0
@test B[1.1,2] === 0.0
@test B[2.1,2] === 1.0
@test_throws BoundsError B[2.1,3]
@test_throws BoundsError B[3.1,2]

@test all(B[[1.1,2.1], 1] .=== [1.0,0.0])
@test all(B[1.1,1:2] .=== [1.0,0.0])
@test all(B[[1.1,2.1], 1:2] .=== [1.0 0.0; 0.0 1.0])

@test_throws BoundsError B[[0.1,2.1], 1]
