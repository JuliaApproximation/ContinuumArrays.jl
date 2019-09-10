abstract type OrthogonalPolynomial{T} <: Basis{T} end



@simplify function \(A::OrthogonalPolynomial, *(B::Identity, C::OrthogonalPolynomial))
    A === C || error("Override  $(typeof(A)) \\ (x * $(typeof(C)))")
    jacobimatrix(A)
end


@simplify *(X::Identity, P::OrthogonalPolynomial) = ApplyQuasiMatrix(*, P, apply(\, P, applied(*, X, P)))
  

function forwardrecurrence!(v::AbstractVector{T}, b::AbstractVector, a::AbstractVector, c::AbstractVector, x) where T
    isempty(v) && return v
    v[1] = one(T) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = (x-a[1])/c[1]
    @inbounds for n=3:length(v)
        v[n] = muladd(x-a[n-1],v[n-1],-b[n-1]*v[n-2])/c[n-1]
    end
    v
end

function forwardrecurrence!(v::AbstractVector{T}, b::AbstractVector, ::Zeros{<:Any,1}, c::AbstractVector, x) where T
    isempty(v) && return v
    v[1] = one(T) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = x/c[1]
    @inbounds for n=3:length(v)
        v[n] = muladd(x,v[n-1],-b[n-1]*v[n-2])/c[n-1]
    end
    v
end

# special case for Chebyshev
function forwardrecurrence!(v::AbstractVector{T}, b::AbstractVector, ::Zeros{<:Any,1}, c::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractVector}}, x) where T
    isempty(v) && return v
    c0,c∞ = c.args
    v[1] = one(T) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = x/c0
    @inbounds for n=3:length(v)
        v[n] = muladd(x,v[n-1],-b[n-2]*v[n-2])/c∞[n-2]
    end
    v
end

function forwardrecurrence!(v::AbstractVector{T}, b_v::AbstractFill, ::Zeros{<:Any,1}, c::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, x) where T
    isempty(v) && return v
    c0,c∞_v = c.args
    b = getindex_value(b_v)
    c∞ = getindex_value(c∞_v) 
    mbc  = -b/c∞
    xc = x/c∞
    v[1] = one(T) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = x/c0
    @inbounds for n=3:length(v)
        v[n] = muladd(xc,v[n-1],mbc*v[n-2])
    end
    v
end

_vec(a) = vec(a)
_vec(a::Adjoint{<:Any,<:AbstractVector}) = a'
bands(J) = _vec.(J.data.args)

function getindex(P::OrthogonalPolynomial{T}, x::Number, n::OneTo) where T
    J = jacobimatrix(P)
    b,a,c = bands(J)
    forwardrecurrence!(similar(n,T),b,a,c,x)
end

function getindex(P::OrthogonalPolynomial{T}, x::AbstractVector, n::OneTo) where T
    J = jacobimatrix(P)
    b,a,c = bands(J)
    V = Matrix{T}(undef,length(x),length(n))
    for k = eachindex(x)
        forwardrecurrence!(view(V,k,:),b,a,c,x[k])
    end
    V
end

getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][n]

getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][:,n]    

getindex(P::OrthogonalPolynomial, x::Number, n::Number) = P[x,OneTo(n)][end]