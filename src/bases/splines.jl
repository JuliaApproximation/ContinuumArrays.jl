struct Spline{order,T,P<:AbstractVector} <: Basis{T}
    points::P
end
Spline{o,T}(pts::P) where {o,T,P} = Spline{o,T,P}(pts)

const LinearSpline = Spline{1}
const HeavisideSpline = Spline{0}

Spline{o}(pts::AbstractVector{T}) where {o,T} = Spline{o,float(T)}(pts)
Spline{o}(S::Spline) where {o} = Spline{o}(S.points)

convert(::Type{AbstractQuasiArray{T}}, S::Spline{λ,T}) where {λ,T} = S
convert(::Type{AbstractQuasiMatrix{T}}, S::Spline{λ,T}) where {λ,T} = S
convert(::Type{AbstractQuasiArray{T}}, S::Spline{λ}) where {λ,T} = Spline{λ,T}(S.points)
convert(::Type{AbstractQuasiMatrix{T}}, S::Spline{λ}) where {λ,T} = convert(AbstractQuasiArray{T}, S)

for Typ in (:LinearSpline, :HeavisideSpline)
    STyp = string(Typ)
    @eval function show(io::IO, L::$Typ)
        print(io, "$($STyp)(")
        print(IOContext(io, :limit=>true), L.points)
        print(io,")")
    end
end

axes(B::Spline{o}) where o =
    (Inclusion(first(B.points)..last(B.points)), OneTo(length(B.points)+o-1))
==(A::Spline{o}, B::Spline{o}) where o = A.points == B.points

function getindex(B::LinearSpline{T}, x::Number, k::Int) where T
    @boundscheck (x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2)) || throw(BoundsError(B, (x, k)))

    p = B.points
    n = length(p)

    k > 1 && x ≤ p[k-1] && return zero(T)
    k < n && x ≥ p[k+1] && return zero(T)
    x == p[k] && return one(T)
    x < p[k] && return (x-p[k-1])/(p[k]-p[k-1])
    return (x-p[k+1])/(p[k]-p[k+1]) # x ≥ p[k]
end

function getindex(B::HeavisideSpline{T}, x::Number, k::Int) where T
    @boundscheck (x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2)) || throw(BoundsError(B, (x, k)))

    p = B.points
    p[k] < x < p[k+1] && return one(T)
    p[k] == x && return one(T)/2
    p[k+1] == x && return one(T)/2
    return zero(T)
end

function getindex(B::Spline{-1,T}, x::Number, k::Int) where T
    @boundscheck (x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2)) || throw(BoundsError(B, (x, k)))

    p = B.points
    p[k+1] == x && return convert(T,Inf)
    zero(T)
end



grid(L::HeavisideSpline, n...) = L.points[1:end-1] .+ diff(L.points)/2
plotgrid(L::HeavisideSpline, n...) = [L.points'; L.points'][2:end-1]
function plotgridvalues(f::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{HeavisideSpline,Any}})
    g = plotgrid(basis(f))
    c = coefficients(f)
    g,vec([c'; c'])
end

function plotgrid(L::Spline{-1}, n...)
    p = L.points[2:end-1]
    vec([p'; p'; p'])
end
function plotgridvalues(f::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{Spline{-1},Any}})
    g = plotgrid(basis(f))
    c = coefficients(f)
    g,vec([zeros(1,length(c)); c'; fill(NaN,1,length(c))])
end


# Splines sample same number of points regardless of length.
grid(L::HeavisideSpline, ::Integer) = L.points[1:end-1] .+ diff(L.points)/2
grid(L::LinearSpline, ::Integer) = L.points

## Sub-bases


## Gram matrix
function grammatrix(A::LinearSpline{T}) where T
    x = A.points; n = length(x)
    dv,ev = Vector{T}(undef, n), Vector{T}(undef, n-1)

    dv[1] = (x[2]-x[1])/3
    @inbounds for k = 2:n-1
        dv[k] = (x[k+1]-x[k-1])/3
    end
    dv[n] = (x[n] - x[n-1])/3

    @inbounds for k = 1:n-1
        ev[k] = (x[k+1]-x[k])/6
    end

    SymTridiagonal(dv, ev)
end


grammatrix(A::HeavisideSpline) = Diagonal(diff(A.points))


## Differentiation
function diff(L::LinearSpline{T}; dims::Integer=1) where T
    dims == 1 || error("not implemented")
    n = size(L,2)
    x = L.points
    D = BandedMatrix{T}(undef, (n-1,n), (0,1))
    d = diff(x)
    D[band(0)] .= inv.((-).(d))
    D[band(1)] .= inv.(d)
    ApplyQuasiMatrix(*, HeavisideSpline{T}(x), D)
end

function diff(L::HeavisideSpline{T}; dims::Integer=1) where T
    dims == 1 || error("not implemented")
    n = size(L,2)
    x = L.points
    D = BandedMatrix{T}(undef, (n-1,n), (0,1))
    d = diff(x)
    D[band(0)] .= -one(T)
    D[band(1)] .= one(T)
    ApplyQuasiMatrix(*, Spline{-1,T}(x), D)
end


##
# sum
##

function _sum(A::HeavisideSpline, dims)
    dims == 1 || error("not implemented")
    permutedims(diff(A.points))
end

function _sum(P::LinearSpline, dims)
    dims == 1 || error("not implemented")
    d = diff(P.points)
    ret = Array{float(eltype(d))}(undef, length(d)+1)
    ret[1] = d[1]/2
    for k = 2:length(d)
        ret[k] = (d[k-1] + d[k])/2
    end
    ret[end] = d[end]/2
    permutedims(ret)
end

function _sum(P::Spline{-1,T}, dims) where T
    dims == 1 || error("not implemented")
    Ones{T}(1, size(P,2))
end

_cumsum(H::HeavisideSpline{T}, dims) where T = LinearSpline(H.points) * tril(Ones{T}(length(H.points),length(H.points)-1) .* diff(H.points)',-1)
_cumsum(S::Spline{-1,T}, dims) where T = HeavisideSpline(S.points) * tril(ones(T,length(S.points)-1,length(S.points)-2),-1)