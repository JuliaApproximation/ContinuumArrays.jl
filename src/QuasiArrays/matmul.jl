
const QuasiArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractQuasiArray{T,p}, <:AbstractArray{V,q}}

const ArrayMulQuasiArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractArray{T,p}, <:AbstractQuasiArray{V,q}}

const QuasiArrayMulQuasiArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractQuasiArray{T,p}, <:AbstractQuasiArray{V,q}}
####
# Matrix * Vector
####
const QuasiMatMulVec{styleA, styleB, T, V} = QuasiArrayMulArray{styleA, styleB, 2, 1, T, V}


function getindex(M::QuasiMatMulVec, k::Real)
    A,B = M.factors
    ret = zero(eltype(M))
    @inbounds for j in axes(A,2)
        ret += A[k,j] * B[j]
    end
    ret
end

function getindex(M::QuasiMatMulVec, k::AbstractArray)
    A,B = M.factors
    ret = zeros(eltype(M),length(k))
    @inbounds for j in axes(A,2)
        ret .+= view(A,k,j) .* B[j]
    end
    ret
end


QuasiMatMulMat{styleA, styleB, T, V} = QuasiArrayMulArray{styleA, styleB, 2, 2, T, V}
QuasiMatMulQuasiMat{styleA, styleB, T, V} = QuasiArrayMulQuasiArray{styleA, styleB, 2, 2, T, V}

*(A::AbstractQuasiArray, B...) = materialize(Mul(A,B...))
*(A::AbstractQuasiArray, B::AbstractQuasiArray, C...) = materialize(Mul(A,B,C...))
*(A::AbstractArray, B::AbstractQuasiArray, C...) = materialize(Mul(A,B,C...))
pinv(A::AbstractQuasiArray) = materialize(PInv(A))
inv(A::AbstractQuasiArray) = materialize(Inv(A))

*(A::AbstractQuasiArray, B::Mul, C...) = materialize(Mul(A, B.factors..., C...))
*(A::Mul, B::AbstractQuasiArray, C...) = materialize(Mul(A.factors..., B, C...))


####
# MulQuasiArray
#####

struct MulQuasiArray{T, N, MUL<:Mul} <: AbstractQuasiArray{T,N}
    mul::MUL
end

const MulQuasiVector{T, MUL<:Mul} = MulQuasiArray{T, 1, MUL}
const MulQuasiMatrix{T, MUL<:Mul} = MulQuasiArray{T, 2, MUL}

const Vec = MulQuasiVector


MulQuasiArray{T,N}(M::MUL) where {T,N,MUL<:Mul} = MulQuasiArray{T,N,MUL}(M)
MulQuasiArray{T}(M::Mul) where {T} = MulQuasiArray{T,ndims(M)}(M)
MulQuasiArray(M::Mul) = MulQuasiArray{eltype(M)}(M)
MulQuasiVector(M::Mul) = MulQuasiVector{eltype(M)}(M)
MulQuasiMatrix(M::Mul) = MulQuasiMatrix{eltype(M)}(M)

MulQuasiArray(factors...) = MulQuasiArray(Mul(factors...))
MulQuasiArray{T}(factors...) where T = MulQuasiArray{T}(Mul(factors...))
MulQuasiArray{T,N}(factors...) where {T,N} = MulQuasiArray{T,N}(Mul(factors...))
MulQuasiVector(factors...) = MulQuasiVector(Mul(factors...))
MulQuasiMatrix(factors...) = MulQuasiMatrix(Mul(factors...))

_MulArray(factors...) = MulQuasiArray(factors...)
_MulArray(factors::AbstractArray...) = MulArray(factors...)

axes(A::MulQuasiArray) = axes(A.mul)
size(A::MulQuasiArray) = map(length, axes(A))

IndexStyle(::MulQuasiArray{<:Any,1}) = IndexLinear()

==(A::MulQuasiArray, B::MulQuasiArray) = A.mul == B.mul

@propagate_inbounds getindex(A::MulQuasiArray, kj::Real...) = A.mul[kj...]

*(A::MulQuasiArray, B::MulQuasiArray) = A.mul * B.mul
*(A::MulQuasiArray, B::Mul) = A.mul * B
*(A::Mul, B::MulQuasiArray) = A * B.mul
*(A::MulQuasiArray, B::AbstractQuasiArray) = A.mul * B
*(A::AbstractQuasiArray, B::MulQuasiArray) = A * B.mul
*(A::MulQuasiArray, B::AbstractArray) = A.mul * B
*(A::AbstractArray, B::MulQuasiArray) = A * B.mul

adjoint(A::MulQuasiArray) = MulQuasiArray(reverse(adjoint.(A.mul.factors))...)
transpose(A::MulQuasiArray) = MulQuasiArray(reverse(transpose.(A.mul.factors))...)


MemoryLayout(M::MulQuasiArray) = MulLayout(MemoryLayout.(M.mul.factors))


## PInvQuasiMatrix

function _PInvQuasiMatrix end

struct PInvQuasiMatrix{T, PINV<:AbstractPInv} <: AbstractQuasiMatrix{T}
    pinv::PINV
    global _PInvQuasiMatrix(pinv::PINV) where PINV = new{eltype(pinv), PINV}(pinv)
end

const InvQuasiMatrix{T, INV<:Inv} = PInvQuasiMatrix{T,INV}

PInvQuasiMatrix(M) = _PInvQuasiMatrix(PInv(M))
InvQuasiMatrix(M) = _PInvQuasiMatrix(Inv(M))

axes(A::PInvQuasiMatrix) = axes(A.pinv)
size(A::PInvQuasiMatrix) = map(length, axes(A))

@propagate_inbounds getindex(A::PInvQuasiMatrix{T}, k::Int, j::Int) where T =
    (A.pinv*[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

MemoryLayout(M::PInvQuasiMatrix) = MemoryLayout(M.pinv)

*(A::PInvQuasiMatrix, B::AbstractQuasiMatrix, C...) = *(A.pinv, B, C...)
*(A::PInvQuasiMatrix, B::MulQuasiArray, C...) = *(A.pinv, B.mul, C...)


####
# Matrix * Array
####

_flatten(A::MulQuasiArray, B...) = _flatten(A.mul.factors..., B...)
flatten(A::MulQuasiArray) = MulQuasiArray(Mul(_flatten(A.mul.factors...)))


# the default is always Array

_materialize(M::QuasiArrayMulArray, _) = MulQuasiArray(M)
_materialize(M::ArrayMulQuasiArray, _) = MulQuasiArray(M)
_materialize(M::QuasiArrayMulQuasiArray, _) = MulQuasiArray(M)



# if multiplying two MulQuasiArrays simplifies the arguments, we materialize,
# otherwise we leave it as a lazy object
_mulquasi_join(As, M::MulQuasiArray, Cs) = MulQuasiArray(As..., M.mul.factors..., Cs...)
_mulquasi_join(As, B, Cs) = *(As..., B, Cs...)


function _materialize(M::Mul2{<:Any,<:Any,<:MulQuasiArray,<:MulQuasiArray}, _)
    As, Bs = M.factors
    _mul_join(reverse(tail(reverse(As))), last(As) * first(Bs), tail(Bs))
end


function _materialize(M::Mul2{<:Any,<:Any,<:MulQuasiArray,<:AbstractQuasiArray}, _)
    As, B = M.factors
    rmaterialize(Mul(As.mul.factors..., B))
end

function _materialize(M::Mul2{<:Any,<:Any,<:AbstractQuasiArray,<:MulQuasiArray}, _)
    A, Bs = M.factors
    *(A, Bs.mul.factors...)
end

# A MulQuasiArray can't be materialized further left-to-right, so we do right-to-left
function _materialize(M::Mul2{<:Any,<:Any,<:MulQuasiArray,<:AbstractArray}, _)
    As, B = M.factors
    rmaterialize(Mul(As.mul.factors..., B))
end

function _lmaterialize(A::MulQuasiArray, B, C...)
    As = A.mul.factors
    flatten(_MulArray(reverse(tail(reverse(As)))..., _lmaterialize(last(As), B, C...)))
end



function _rmaterialize(Z::MulQuasiArray, Y, W...)
    Zs = Z.mul.factors
    flatten(_MulArray(_rmaterialize(first(Zs), Y, W...), tail(Zs)...))
end
