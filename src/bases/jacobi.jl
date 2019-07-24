abstract type AbstractJacobiWeight{T} <: AbstractQuasiVector{T} end

struct JacobiWeight{T} <: AbstractJacobiWeight{T}
    b::T
    a::T
    JacobiWeight{T}(b, a) where T = new{T}(convert(T,b), convert(T,a))
end

JacobiWeight(b::T, a::V) where {T,V} = JacobiWeight{promote_type(T,V)}(b,a)

axes(::AbstractJacobiWeight) = (Inclusion(ChebyshevInterval()),)
function getindex(w::JacobiWeight, x::Real)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x)^w.a * (1+x)^w.b
end



abstract type AbstractJacobi{T} <: OrthogonalPolynomial{T} end

struct Legendre{T} <: AbstractJacobi{T} end
Legendre() = Legendre{Float64}()

==(::Legendre, ::Legendre) = true

struct Jacobi{T} <: AbstractJacobi{T}
    b::T
    a::T
    Jacobi{T}(b, a) where T = new{T}(convert(T,b), convert(T,a))
end

Jacobi(b::T, a::V) where {T,V} = Jacobi{promote_type(T,V)}(b,a)

axes(::AbstractJacobi) = (Inclusion(ChebyshevInterval()), OneTo(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b

########
# Mass Matrix
#########

materialize(M::QMul2{<:QuasiAdjoint{<:Any,<:Legendre},<:Legendre}) =
    Diagonal(2 ./ (2(0:∞) .+ 1))

########
# Jacobi Matrix
########

materialize(M::Ldiv{<:Any,<:Legendre,
                               <:QMul2{<:Identity,
                                       <:Legendre}}) =
    _BandedMatrix(Vcat(((0:∞)./(1:2:∞))', Zeros(1,∞), ((1:∞)./(1:2:∞))'), ∞, 1,1)


function materialize(M::QMul2{<:Identity,<:Legendre})
    X, P = M.args
    MulQuasiMatrix(P, pinv(P)*X*P)
end



##########
# Conversion
##########

function materialize(M::Ldiv{BasisStyle,<:Jacobi,<:Jacobi}) 
    A,B = M.args
    a,b = B.a,B.b
    if A.a == a && A.b == b+1
        _BandedMatrix(Vcat((((0:∞) .+ a)./((1:2:∞) .+ (a+b)))', (((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)))'), ∞, 0,1)
    end
end

##########
# Derivatives
##########

# Jacobi(b+1,a+1)\(D*Jacobi(a,b))
function materialize(M::Ldiv{BasisStyle,<:Jacobi,
                                        <:QMul2{<:Derivative{<:Any,<:ChebyshevInterval},
                                                <:Jacobi}})
    J, DS = M.args
    D,S = DS.args
    (J.b == S.b+1 && J.a == S.a+1) || throw(ArgumentError())
    _BandedMatrix((((1:∞) .+ (S.a + S.b))/2)', ∞, -1,1)
end


function materialize(M::QMul2{<:Derivative{<:Any,<:ChebyshevInterval},<:Jacobi})
    D, S = M.args
    A = apply(\,Jacobi(S.b+1,S.a+1),applied(*,D,S))
    MulQuasiMatrix(Jacobi(S.b+1,S.a+1), A)
end

# Legendre()\ (D*W*Jacobi(true,true))
function materialize(M::Ldiv{BasisStyle,<:Legendre,
                                        QMul3{<:Derivative{<:Any,<:ChebyshevInterval},
                                              QuasiDiagonal{Bool,JacobiWeight{Bool}},
                                              Jacobi{Bool}}})
    L, DWS = M.args
    D, W, S = DWS.args
    w = parent(W)
    (w.a && S.a && w.b && S.b) || throw(ArgumentError())
    _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
end

# reduce to Legendre
function materialize(M::QMul3{<:Derivative{<:Any,<:ChebyshevInterval},
                              QuasiDiagonal{Bool,JacobiWeight{Bool}},
                              Jacobi{Bool}})
    D, W, S = M.args
    w = parent(W)
    (w.a && S.a && w.b && S.b) || throw(ArgumentError())
    A = apply(\, Legendre{eltype(M)}(), applied(*,D,W,S))
    MulQuasiMatrix(Legendre(), A)
end

# Jacobi(b-1,a-1)\ (D*w*Jacobi(b,a))
function materialize(M::Ldiv{BasisStyle,<:Jacobi,
    QMul3{<:Derivative{<:Any,<:ChebyshevInterval},
          <:QuasiDiagonal{<:Any,<:JacobiWeight},
          <:Jacobi}})
    L, DWS = M.args
    D, W, S = DWS.args
    w = parent(W)
    (w.a == S.a == L.a+1 && w.b == S.b == L.b+1) || throw(ArgumentError())
    _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
end

function materialize(M::Ldiv{BasisStyle,<:Jacobi{Bool},
                                    <:QMul2{QuasiDiagonal{Bool,JacobiWeight{Bool}},
                                            Jacobi{Bool}}})
    J, WS = M.args
    W,S = WS.args
    w = parent(W)
    @assert  S.b && S.a
    if w.b && !w.a
        @assert !J.b && J.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))',((2:2:∞)./(3:2:∞))'), ∞, 1,0)
    elseif !w.b && w.a
        @assert J.b && !J.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))',(-(2:2:∞)./(3:2:∞))'), ∞, 1,0)
    else
        error("Not implemented")
    end
end

function materialize(M::Ldiv{BasisStyle,<:Legendre,
                                        <:QMul2{QuasiDiagonal{Bool,JacobiWeight{Bool}},Jacobi{Bool}}})
    L, WS = M.args
    W,S = WS.args
    w = parent(W)
    if w.b && w.a
        @assert S.b && S.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))', Zeros(1,∞), (-(2:2:∞)./(3:2:∞))'), ∞, 2,0)
    elseif w.b && !w.a
        @assert S.b && !S.a
        _BandedMatrix(Ones{eltype(M)}(2,∞), ∞, 1,0)
    elseif !w.b && w.a
        @assert !S.b && S.a
        _BandedMatrix(Vcat(Ones{eltype(M)}(1,∞),-Ones{eltype(M)}(1,∞)), ∞, 1,0)
    else
        error("Not implemented")
    end
end

function materialize(M::QMul2{QuasiAdjoint{Bool,Jacobi{Bool}},
                                        QuasiDiagonal{Int,JacobiWeight{Int}},Jacobi{Bool}})
    St, W, S = M.args

    w = parent(W)
    (w.b == 2 && S.b && w.a == 2 && S.a && parent(St) == S) || throw(ArgumentError())
    W_sqrt = Diagonal(JacobiWeight(true,true))
    L = Legendre()
    A = PInv(L)*W_sqrt*S
    A'*(L'L)*A
end
