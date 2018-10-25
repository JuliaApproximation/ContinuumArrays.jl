
const QuasiArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractQuasiArray{T,p}, <:AbstractArray{V,q}}

const QuasiArrayMulQuasiArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractQuasiArray{T,p}, <:AbstractQuasiArray{V,q}}
####
# Matrix * Vector
####
const QuasiMatMulVec{styleA, styleB, T, V} = QuasiArrayMulArray{styleA, styleB, 2, 1, T, V}


function getindex(M::QuasiMatMulVec, k::Real)
    A,B = M.factors
    ret = zero(eltype(M))
    @inbounds for j = 1:size(A,2)
        ret += A[k,j] * B[j]
    end
    ret
end

QuasiMatMulMat{styleA, styleB, T, V} = QuasiArrayMulArray{styleA, styleB, 2, 2, T, V}
QuasiMatMulQuasiMat{styleA, styleB, T, V} = QuasiArrayMulQuasiArray{styleA, styleB, 2, 2, T, V}


*(A::AbstractQuasiArray, B::AbstractQuasiArray) = materialize(Mul(A,B))
*(A::AbstractQuasiArray, B::AbstractArray) = materialize(Mul(A,B))
*(A::AbstractArray, B::AbstractQuasiArray) = materialize(Mul(A,B))
inv(A::AbstractQuasiArray) = materialize(Inv(A))
*(A::Inv{<:Any,<:AbstractQuasiArray}, B::AbstractQuasiArray) = materialize(Mul(A,B))


_Mul(A::Mul, B::Mul) = Mul(A.factors..., B.factors...)
_Mul(A::Mul, B) = Mul(A.factors..., B)
_Mul(A, B::Mul) = Mul(A, B.factors...)
_Mul(A, B) = Mul(A, B)
_lsimplify2(A, B...) = _Mul(A, _lsimplify(B...))
_lsimplify2(A::Mul, B...) = _lsimplify2(A.factors..., B...)
_lsimplify(A) = materialize(A)
_lsimplify(A, B) = materialize(Mul(A,B))
_lsimplify(A, B, C, D...) = _lsimplify2(materialize(Mul(A,B)), C, D...)
lsimplify(M::Mul) = _lsimplify(M.factors...)

*(A::AbstractQuasiArray, B::Mul) = lsimplify(_Mul(A, B))
