using ContinuumArrays, LazyArrays, IntervalSets, FillArrays, LinearAlgebra, Test
import ContinuumArrays: AbstractAxisMatrix, ℵ₀, materialize
import Base: axes, getindex, convert

struct Spline{order,T} <: AbstractAxisMatrix{T}
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


## tests

H = HeavisideSpline([1,2,3])
@test size(H) == (ℵ₀, 2)

@test_throws BoundsError H[0.1, 1]
@test H[1.1,1] === H'[1,1.1] === transpose(H)[1,1.1] === 1.0
@test H[2.1,1] === H'[1,2.1] === transpose(H)[1,2.1] === 0.0
@test H[1.1,2] === 0.0
@test H[2.1,2] === 1.0
@test_throws BoundsError H[2.1,3]
@test_throws BoundsError H'[3,2.1]
@test_throws BoundsError transpose(H)[3,2.1]
@test_throws BoundsError H[3.1,2]

@test all(H[[1.1,2.1], 1] .=== H'[1,[1.1,2.1]] .=== transpose(H)[1,[1.1,2.1]] .=== [1.0,0.0])
@test all(H[1.1,1:2] .=== [1.0,0.0])
@test all(H[[1.1,2.1], 1:2] .=== [1.0 0.0; 0.0 1.0])

@test_throws BoundsError H[[0.1,2.1], 1]


L = LinearSpline([1,2,3])
@test size(L) == (ℵ₀, 3)

@test_throws BoundsError L[0.1, 1]
@test L[1.1,1] == L'[1,1.1] == transpose(L)[1,1.1] ≈ 0.9
@test L[2.1,1] === L'[1,2.1] === transpose(L)[1,2.1] === 0.0
@test L[1.1,2] ≈ 0.1
@test L[2.1,2] ≈ 0.9
@test L[2.1,3] == L'[3,2.1] == transpose(L)[3,2.1] ≈ 0.1
@test_throws BoundsError L[3.1,2]
L[[1.1,2.1], 1]
@test L[[1.1,2.1], 1] == L'[1,[1.1,2.1]] == transpose(L)[1,[1.1,2.1]] ≈ [0.9,0.0]
@test L[1.1,1:2] ≈ [0.9,0.1]
@test L[[1.1,2.1], 1:2] ≈ [0.9 0.1; 0.0 0.9]

@test_throws BoundsError L[[0.1,2.1], 1]


f = H*[1,2]
@test axes(f) == (1.0..3.0,)
@test f[1.1] ≈ 1
@test f[2.1] ≈ 2

f = L*[1,2,4]
@test axes(f) == (1.0..3.0,)
@test f[1.1] ≈ 1.1
@test f[2.1] ≈ 2.2

@test H'H == Eye(2)
@test L'L == SymTridiagonal([1/3,2/3,1/3], [1/6,1/6])
