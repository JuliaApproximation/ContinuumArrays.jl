struct JacobiWeight{T} <: AbstractQuasiVector{T}
    b::T
    a::T
end

axes(::JacobiWeight) = (ChebyshevInterval(),)
function getindex(w::JacobiWeight, x::Real)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x)^w.a * (1+x)^w.b
end

abstract type AbstractJacobi{T} <: AbstractQuasiMatrix{T} end

pinv(J::AbstractJacobi) = PInv(J)

struct Legendre{T} <: AbstractJacobi{T} end
Legendre() = Legendre{Float64}()

struct Jacobi{T} <: AbstractJacobi{T}
    b::T
    a::T
end

axes(::AbstractJacobi) = (ChebyshevInterval(), OneTo(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b


materialize(M::Mul2{<:Any,<:Any,<:QuasiAdjoint{<:Any,<:Legendre},<:Legendre}) =
    Diagonal(2 ./ (2(0:∞) .+ 1))

# pinv(Jacobi(b+1,a+1))D*W*Jacobi(a,b)
function materialize(M::Mul{<:Tuple,<:Tuple{<:PInv{<:Any,<:Jacobi},
                                        <:Derivative{<:Any,<:ChebyshevInterval},
                                        <:Jacobi}})
    Ji, D, S = M.factors
    J = parent(Ji)
    (J.b == S.b+1 && J.a == S.a+1) || throw(ArgumentError())
    _BandedMatrix(((1:∞ .+ 1 .+ S.a .+ S.b)/2)', ∞, -1,1)
end


function materialize(M::Mul2{<:Any,<:Any,<:Derivative{<:Any,<:ChebyshevInterval},<:Jacobi})
    D, S = M.factors
    A = PInv(Jacobi(S.b+1,S.a+1))*D*S
    MulQuasiMatrix(Jacobi(S.b+1,S.a+1), A)
end

# pinv(Legendre())D*W*Jacobi(true,true)
function materialize(M::Mul{<:Tuple,<:Tuple{<:PInv{<:Any,<:Legendre},
                                        <:Derivative{<:Any,<:ChebyshevInterval},
                                        QuasiDiagonal{Bool,JacobiWeight{Bool}},Jacobi{Bool}}})
    Li, _, W, S = M.factors
    w = parent(W)
    (w.a && S.a && w.b && S.b) || throw(ArgumentError())
    _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
end

# reduce to Legendre
function materialize(M::Mul{<:Tuple,<:Tuple{<:Derivative{<:Any,<:ChebyshevInterval},
                                        QuasiDiagonal{Bool,JacobiWeight{Bool}},Jacobi{Bool}}})
    D, W, S = M.factors
    w = parent(W)
    (w.a && S.a && w.b && S.b) || throw(ArgumentError())
    A = pinv(Legendre{eltype(M)}())*D*W*S
    MulQuasiMatrix(Legendre(), A)
end

function materialize(M::Mul{<:Tuple,<:Tuple{<:PInv{<:Any,<:Jacobi{Bool}},
                            QuasiDiagonal{Bool,JacobiWeight{Bool}},Jacobi{Bool}}})
    Ji, W, S = M.factors
    J,w = parent(Ji),parent(W)
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

function materialize(M::Mul{<:Tuple,<:Tuple{<:PInv{<:Any,<:Legendre},
                            QuasiDiagonal{Bool,JacobiWeight{Bool}},Jacobi{Bool}}})
    Li, W, S = M.factors
    L,w = parent(Li),parent(W)
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

function materialize(M::Mul{<:Tuple,<:Tuple{QuasiAdjoint{Bool,Jacobi{Bool}},
                                        QuasiDiagonal{Int,JacobiWeight{Int}},Jacobi{Bool}}})
    St, W, S = M.factors

    w = parent(W)
    (w.b == 2 && S.b && w.a == 2 && S.a && parent(St) == S) || throw(ArgumentError())
    W_sqrt = Diagonal(JacobiWeight(true,true))
    L = Legendre()
    A = PInv(L)*W_sqrt*S
    A'*(L'L)*A
end
