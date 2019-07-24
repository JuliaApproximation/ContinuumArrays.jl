struct ChebyshevWeight{T} <: AbstractJacobiWeight{T} end

function getindex(w::ChebyshevWeight, x::Real)
    x ∈ axes(w,1) || throw(BoundsError())
    1/sqrt(1-x^2)
end

struct UltrasphericalWeight{T,Λ} <: AbstractJacobiWeight{T} 
    λ::Λ
end

function getindex(w::UltrasphericalWeight, x::Real)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x^2)^(w.λ-one(w.λ)/2)
end


struct Chebyshev{T} <: AbstractJacobi{T} end
Chebyshev() = Chebyshev{Float64}()
==(a::Chebyshev, b::Chebyshev) = true

struct Ultraspherical{T,Λ} <: AbstractJacobi{T} 
    λ::Λ
end
Ultraspherical{T}(λ::Λ) where {T,Λ} = Ultraspherical{T,Λ}(λ)
Ultraspherical(λ::Λ) where Λ = Ultraspherical{Float64,Λ}(λ)

==(a::Ultraspherical, b::Ultraspherical) = a.λ == b.λ


########
# Jacobi Matrix
########

function materialize(M::Ldiv{BasisStyle,<:Chebyshev,
                               <:QMul2{<:Identity,
                                       <:Chebyshev}}) 
    T = eltype(M)                                       
    _BandedMatrix(Vcat(Fill(one(T)/2,1,∞), Zeros(1,∞), Hcat(one(T), Fill(one(T)/2,1,∞))), ∞, 1, 1)
end

function materialize(M::QMul2{<:Identity,<:Chebyshev})
    X, P = M.args
    MulQuasiMatrix(P, apply(\, P, applied(*, X, P)))
end

function materialize(M::Ldiv{BasisStyle,<:Ultraspherical,
                               <:QMul2{<:Identity,
                                       <:Ultraspherical}}) 
    T = eltype(M)         
    P,_ = M.args                              
    λ = P.λ
    _BandedMatrix(Vcat((((2λ-1):∞) ./ (2 .*((zero(T):∞) .+ λ)))',
                        Zeros{T}(1,∞),
                        ((one(T):∞) ./ (2 .*((zero(T):∞) .+ λ)))'), ∞, 1, 1)
end


function materialize(M::QMul2{<:Identity,<:Ultraspherical})
    X, P = M.args
    MulQuasiMatrix(P, apply(\, P, applied(*, X, P)))
end


##########
# Derivatives
##########

# Ultraspherical(1)\(D*Chebyshev())
function materialize(M::Ldiv{BasisStyle,<:Ultraspherical,
    <:QMul2{<:Derivative{<:Any,<:ChebyshevInterval},
            <:Chebyshev}})
    J, DS = M.args
    D,S = DS.args
    (J.λ == 1) || throw(ArgumentError())
    _BandedMatrix((zero(eltype(M)):∞)', ∞, -1,1)
end


function materialize(M::QMul2{<:Derivative{<:Any,<:ChebyshevInterval},<:Chebyshev})
    D, S = M.args
    A = apply(\,Ultraspherical{eltype(S)}(1),applied(*,D,S))
    MulQuasiMatrix(Ultraspherical{eltype(S)}(1), A)
end

# Ultraspherical(λ+1)\(D*Ultraspherical(λ))
function materialize(M::Ldiv{BasisStyle,<:Ultraspherical,
    <:QMul2{<:Derivative{<:Any,<:ChebyshevInterval},
            <:Ultraspherical}})
    J, DS = M.args
    D,S = DS.args
    (J.λ == S.λ+1) || throw(ArgumentError())
    _BandedMatrix(Fill(2S.λ,1,∞), ∞, -1,1)
end


function materialize(M::QMul2{<:Derivative{<:Any,<:ChebyshevInterval},<:Ultraspherical})
    D, S = M.args
    A = apply(\,Ultraspherical{eltype(S)}(S.λ+1),applied(*,D,S))
    MulQuasiMatrix(Ultraspherical{eltype(S)}(S.λ+1), A)
end


##########
# Converision
##########


function materialize(M::Ldiv{BasisStyle,<:Ultraspherical,<:Chebyshev})
    U,C = M.args
    (U.λ == 1) || throw(ArgumentError())
    T = eltype(M)
    BandedMatrix(0 => Vcat([one(T)],Fill(one(T)/2,∞)), 2 => Vcat([-one(T)/2],Fill(-one(T)/2,∞)))
end

function materialize(M::Ldiv{BasisStyle,<:Ultraspherical,<:Ultraspherical})
    C2,C1 = M.args
    λ = C1.λ
    T = eltype(M)
    if C2.λ == λ+1 
        _BandedMatrix( Vcat(-(λ ./ (1:∞ .+ λ))', Zeros(1,∞), (λ ./ (1:∞ .+ λ))'), ∞, 0, 2)
    elseif C2.λ == λ
        Eye{T}(∞)
    else
        throw(ArgumentError())
    end
end


####
# interrelationships
####

# (18.7.3)

function materialize(M::Ldiv{BasisStyle,<:Chebyshev,<:Jacobi})
    A,B = M.args
    T = eltype(M)
    (B.a == B.b == -T/2) || throw(ArgumentError())
    Diagonal(Jacobi(-T/2,-T/2)[1,:])
end