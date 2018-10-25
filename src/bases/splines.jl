
struct Spline{order,T} <: AbstractQuasiMatrix{T}
    points::Vector{T}
end

const LinearSpline{T} = Spline{1,T}
const HeavisideSpline{T} = Spline{0,T}

Spline{o}(pts::AbstractVector{T}) where {o,T} = Spline{o,float(T)}(pts)

axes(B::Spline{o}) where o = (first(B.points)..last(B.points), Base.OneTo(length(B.points)+o-1))

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


function convert(::Type{SymTridiagonal}, AB::Mul{T,<:Any,<:Any,<:ContinuumArrays.Adjoint{<:Any,<:LinearSpline},<:LinearSpline}) where T
    Ac,B = AB.A, AB.B
    A = parent(Ac)
    @assert A.points == B.points
    x = A.points; n = length(x)
    dv = Vector{T}(undef, n)
    dv[1] = (x[2]-x[1])/3
    for k = 2:n-1
        dv[k] = (x[k+1]-x[k-1])/3
    end
    dv[n] = (x[n] - x[n-1])/3

    SymTridiagonal(dv, diff(x)./6)
end
#
materialize(M::Mul{<:Any,<:Any,<:Any,<:ContinuumArrays.Adjoint{<:Any,<:LinearSpline},<:LinearSpline}) =
    convert(SymTridiagonal, M)

function materialize(M::Mul{T,<:Any,<:Any,<:ContinuumArrays.Adjoint{<:Any,<:HeavisideSpline},<:HeavisideSpline}) where T
    Ac, B = M.A, M.B
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A.points == B.points || throw(ArgumentError("Cannot multiply incompatible splines"))
    Diagonal(diff(A.points))
end
