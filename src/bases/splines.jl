struct Spline{order,T,P<:AbstractVector} <: Basis{T}
    points::P
end
Spline{o,T}(pts::P) where {o,T,P} = Spline{o,T,P}(pts)

const LinearSpline = Spline{1}
const HeavisideSpline = Spline{0}

Spline{o}(pts::AbstractVector{T}) where {o,T} = Spline{o,float(T)}(pts)

axes(B::Spline{o}) where o =
    (Inclusion(first(B.points)..last(B.points)), OneTo(length(B.points)+o-1))
==(A::Spline{o}, B::Spline{o}) where o = A.points == B.points

function getindex(B::LinearSpline{T}, x::Number, k::Int) where T
    x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2)|| throw(BoundsError())

    p = B.points
    n = length(p)

    k > 1 && x ≤ p[k-1] && return zero(T)
    k < n && x ≥ p[k+1] && return zero(T)
    x == p[k] && return one(T)
    x < p[k] && return (x-p[k-1])/(p[k]-p[k-1])
    return (x-p[k+1])/(p[k]-p[k+1]) # x ≥ p[k]
end

function getindex(B::HeavisideSpline{T}, x::Number, k::Int) where T
    x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2)|| throw(BoundsError())

    p = B.points
    n = length(p)

    p[k] < x < p[k+1] && return one(T)
    p[k] == x && return one(T)/2
    p[k+1] == x && return one(T)/2
    return zero(T)
end

grid(L::LinearSpline) = L.points
transform(L::LinearSpline{T}) where T = grid(L),Eye{T}(size(L,2))
function transform(V::SubQuasiArray{<:Any,2,<:LinearSpline,<:Tuple{<:Inclusion,<:Any}})
    g, T = transform(parent(V))
    g, qr(layout_getindex(T,:,parentindices(V)[2])) # avoid sparse matrices of sub-diagonal
end

## Sub-bases


## Mass matrix
function similar(AB::QMul2{<:QuasiAdjoint{<:Any,<:LinearSpline},<:LinearSpline}, ::Type{T}) where T
    n = size(AB,1)
    SymTridiagonal(Vector{T}(undef, n), Vector{T}(undef, n-1))
end
#
@simplify function *(Ac::QuasiAdjoint{<:Any,<:LinearSpline}, B::LinearSpline) 
    M = Mul(Ac, B)
    copyto!(similar(M, eltype(M)), M)
end

function copyto!(dest::SymTridiagonal,
                 AB::QMul2{<:QuasiAdjoint{<:Any,<:LinearSpline},<:LinearSpline}) where T
    Ac,B = AB.A,AB.B
    A = parent(Ac)
    A.points == B.points || throw(ArgumentError())
    dv,ev = dest.dv,dest.ev
    x = A.points; n = length(x)
    length(dv) == n || throw(DimensionMismatch())

    dv[1] = (x[2]-x[1])/3
    @inbounds for k = 2:n-1
        dv[k] = (x[k+1]-x[k-1])/3
    end
    dv[n] = (x[n] - x[n-1])/3

    @inbounds for k = 1:n-1
        ev[k] = (x[k+1]-x[k])/6
    end

    dest
end


@simplify function *(Ac::QuasiAdjoint{<:Any,<:HeavisideSpline}, B::HeavisideSpline)
    A = parent(Ac)
    A.points == B.points || throw(ArgumentError("Cannot multiply incompatible splines"))
    Diagonal(diff(A.points))
end


## Derivative
function copyto!(dest::MulQuasiMatrix{<:Any,<:Tuple{<:HeavisideSpline,<:Any}},
                 M::QMul2{<:Derivative,<:LinearSpline})
    D, L = M.A, M.B
    H, A = dest.args
    x = H.points

    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))
    x == L.points || throw(ArgumentError("Cannot multiply incompatible splines"))
    bandwidths(A) == (0,1) || throw(ArgumentError("Not implemented"))

    d = diff(x)
    A[band(0)] .= inv.((-).(d))
    A[band(1)] .= inv.(d)

    dest
end

function similar(M::QMul2{<:Derivative,<:LinearSpline}, ::Type{T}) where T
    D, B = M.A, M.B
    n = size(B,2)
    ApplyQuasiMatrix(*, HeavisideSpline{T}(B.points),
        BandedMatrix{T}(undef, (n-1,n), (0,1)))
end

@simplify function *(D::Derivative, L::LinearSpline)
    M = Mul(D, L)
    copyto!(similar(M, eltype(M)), M)
end


##
# sum
##

function _sum(A::HeavisideSpline, dims)
    @assert dims == 1
    permutedims(diff(A.points))
end