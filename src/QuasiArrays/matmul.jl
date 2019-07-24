
const QuasiArrayMulArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractQuasiArray{T,p}, <:AbstractArray{V,q}}}

const ArrayMulQuasiArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractArray{T,p}, <:AbstractQuasiArray{V,q}}}

const QuasiArrayMulQuasiArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractQuasiArray{T,p}, <:AbstractQuasiArray{V,q}}}
####
# Matrix * Vector
####
const QuasiMatMulVec{T, V} = QuasiArrayMulArray{2, 1, T, V}
const QuasiMatMulMat{T, V} = QuasiArrayMulArray{2, 2, T, V}
const QuasiMatMulQuasiMat{T, V} = QuasiArrayMulQuasiArray{2, 2, T, V}


import LazyArrays: _mul, rowsupport

function getindex(M::Mul, k::Real)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    for j = rowsupport(A, k) ∩ colsupport(B,1)
        ret += A[k,j] * B[j]
    end
    ret
end

function getindex(M::Mul, k::Real, j::Real)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end


function getindex(M::QuasiMatMulVec, k::Real)
    A,B = M.args
    ret = zero(eltype(M))
    @inbounds for j in axes(A,2)
        ret += A[k,j] * B[j]
    end
    ret
end

function getindex(M::QuasiMatMulVec, k::AbstractArray)
    A,B = M.args
    ret = zeros(eltype(M),length(k))
    @inbounds for j in axes(A,2)
        ret .+= view(A,k,j) .* B[j]
    end
    ret
end


# Used for when a lazy version should be constructed on materialize
abstract type AbstractQuasiArrayApplyStyle <: ApplyStyle end
struct LazyQuasiArrayApplyStyle <: AbstractQuasiArrayApplyStyle end
struct QuasiArrayApplyStyle <: AbstractQuasiArrayApplyStyle end

ndims(M::Applied{LazyQuasiArrayApplyStyle,typeof(*)}) = ndims(last(M.args))


*(A::AbstractQuasiArray, B...) = fullmaterialize(apply(*,A,B...))
*(A::AbstractQuasiArray, B::AbstractQuasiArray, C...) = fullmaterialize(apply(*,A,B,C...))
*(A::AbstractArray, B::AbstractQuasiArray, C...) = fullmaterialize(apply(*,A,B,C...))

pinv(A::AbstractQuasiArray) = materialize(PInv(A))
inv(A::AbstractQuasiArray) = materialize(Inv(A))

axes(L::Ldiv{<:Any,<:Any,<:AbstractQuasiMatrix}) =
    (axes(L.args[1], 2),axes(L.args[2],2))
axes(L::Ldiv{<:Any,<:Any,<:AbstractQuasiVector}) =
    (axes(L.args[1], 2),)    

 \(A::AbstractQuasiArray, B::AbstractQuasiArray) = materialize(Ldiv(A,B))


*(A::AbstractQuasiArray, B::Mul, C...) = fullmaterialize(apply(*,A, B.args..., C...))
*(A::Mul, B::AbstractQuasiArray, C...) = fullmaterialize(apply(*,A.args..., B, C...))


struct ApplyQuasiArray{T, N, App<:Applied} <: AbstractQuasiArray{T,N}
    applied::App
end

const ApplyQuasiVector{T, App<:Applied} = ApplyQuasiArray{T, 1, App}
const ApplyQuasiMatrix{T, App<:Applied} = ApplyQuasiArray{T, 2, App}


ApplyQuasiArray{T,N}(M::App) where {T,N,App<:Applied} = ApplyQuasiArray{T,N,App}(M)
ApplyQuasiArray{T}(M::Applied) where {T} = ApplyQuasiArray{T,ndims(M)}(M)
ApplyQuasiArray(M::Applied) = ApplyQuasiArray{eltype(M)}(M)
ApplyQuasiVector(M::Applied) = ApplyQuasiVector{eltype(M)}(M)
ApplyQuasiMatrix(M::Applied) = ApplyQuasiMatrix{eltype(M)}(M)

ApplyQuasiArray(f, factors...) = ApplyQuasiArray(applied(f, factors...))
ApplyQuasiArray{T}(f, factors...) where T = ApplyQuasiArray{T}(applied(f, factors...))
ApplyQuasiArray{T,N}(f, factors...) where {T,N} = ApplyQuasiArray{T,N}(applied(f, factors...))

ApplyQuasiVector(f, factors...) = ApplyQuasiVector(applied(f, factors...))
ApplyQuasiMatrix(f, factors...) = ApplyQuasiMatrix(applied(f, factors...))

axes(A::ApplyQuasiArray) = axes(A.applied)
size(A::ApplyQuasiArray) = map(length, axes(A))

IndexStyle(::ApplyQuasiArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::ApplyQuasiArray{T,N}, kj::Vararg{Real,N}) where {T,N} =
    A.applied[kj...]

MemoryLayout(M::ApplyQuasiArray) = ApplyLayout(M.applied.f, MemoryLayout.(M.applied.args))

materialize(A::Applied{LazyQuasiArrayApplyStyle}) = ApplyQuasiArray(A)

@inline copyto!(dest::AbstractQuasiArray, M::Applied) = _copyto!(MemoryLayout(dest), dest, M)
@inline _copyto!(_, dest::AbstractQuasiArray, M::Applied) = copyto!(dest, materialize(M))


####
# MulQuasiArray
#####

const MulQuasiArray{T, N, MUL<:Mul} = ApplyQuasiArray{T, N, MUL}

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

most(a) = reverse(tail(reverse(a)))

MulQuasiOrArray = Union{MulArray,MulQuasiArray}

_factors(M::MulQuasiOrArray) = M.applied.args
_factors(M) = (M,)

_flatten(A::MulQuasiArray, B...) = _flatten(A.applied, B...)
flatten(A::MulQuasiArray) = MulQuasiArray(flatten(A.applied))


function fullmaterialize(M::Applied{<:Any,typeof(*)})
    M_mat = materialize(flatten(M))
    typeof(M_mat) <: MulQuasiOrArray || return M_mat
    typeof(M_mat.applied) == typeof(M) || return(fullmaterialize(M_mat))

    ABC = M_mat.applied.args
    length(ABC) ≤ 2 && return flatten(M_mat)

    AB = most(ABC)
    Mhead = fullmaterialize(Mul(AB...))

    typeof(_factors(Mhead)) == typeof(AB) ||
        return fullmaterialize(Mul(_factors(Mhead)..., last(ABC)))

    BC = tail(ABC)
    Mtail =  fullmaterialize(Mul(BC...))
    typeof(_factors(Mtail)) == typeof(BC) ||
        return fullmaterialize(Mul(first(ABC), _factors(Mtail)...))

    apply(*,first(ABC), Mtail.applied.args...)
end

fullmaterialize(M::ApplyQuasiArray) = flatten(fullmaterialize(M.applied))
fullmaterialize(M) = flatten(M)

*(A::MulQuasiArray, B::MulQuasiArray) = flatten(fullmaterialize(apply(*,A.applied.args..., B.applied.args...)))
*(A::MulQuasiArray, B::AbstractQuasiArray) = flatten(fullmaterialize(apply(*,A.applied.args..., B)))
*(A::AbstractQuasiArray, B::MulQuasiArray) = flatten(fullmaterialize(apply(*,A, B.applied.args...)))
*(A::MulQuasiArray, B::AbstractArray) = flatten(fullmaterialize(apply(*,A.applied.args..., B)))
*(A::AbstractArray, B::MulQuasiArray) = flatten(fullmaterialize(apply(*,A, B.applied.args...)))



adjoint(A::MulQuasiArray) = MulQuasiArray(reverse(adjoint.(A.applied.args))...)
transpose(A::MulQuasiArray) = MulQuasiArray(reverse(transpose.(A.applied.args))...)

function similar(A::MulQuasiArray)
    B,a = A.applied.args
    B*similar(a)
end

function similar(A::QuasiArrayMulArray)
    B,a = A.args
    applied(*, B, similar(a))
end

function copy(a::MulQuasiArray)
    @_propagate_inbounds_meta
    copymutable(a)
end

function copyto!(dest::MulQuasiArray, src::MulQuasiArray)
    d = last(dest.applied.args)
    s = last(src.applied.args)
    copyto!(IndexStyle(d), d, IndexStyle(s), s)
    dest
end


MemoryLayout(M::MulQuasiArray) = MulLayout(MemoryLayout.(M.applied.args))

ApplyStyle(::typeof(*), ::AbstractQuasiArray, B...) =
    LazyQuasiArrayApplyStyle()
ApplyStyle(::typeof(*), ::AbstractArray, ::AbstractQuasiArray, B...) =
    LazyQuasiArrayApplyStyle()
ApplyStyle(::typeof(*), ::AbstractArray, ::AbstractArray, ::AbstractQuasiArray, B...) =
    LazyQuasiArrayApplyStyle()

ApplyStyle(::typeof(\), ::AbstractQuasiArray, ::AbstractQuasiArray) =
    LazyQuasiArrayApplyStyle()
ApplyStyle(::typeof(\), ::AbstractQuasiArray, ::AbstractArray) =
    LazyQuasiArrayApplyStyle()
ApplyStyle(::typeof(\), ::AbstractArray, ::AbstractQuasiArray) =
    LazyQuasiArrayApplyStyle()    

for op in (:pinv, :inv)
    @eval ApplyStyle(::typeof($op), args::AbstractQuasiArray) =
        LazyQuasiArrayApplyStyle()
end
## PInvQuasiMatrix


const PInvQuasiMatrix{T, PINV<:InvOrPInv} = ApplyQuasiMatrix{T,PINV}
const InvQuasiMatrix{T, INV<:Inv} = PInvQuasiMatrix{T,INV}

PInvQuasiMatrix(M) = _PInvQuasiMatrix(PInv(M))
InvQuasiMatrix(M) = _PInvQuasiMatrix(Inv(M))

axes(A::PInvQuasiMatrix) = axes(A.applied)
size(A::PInvQuasiMatrix) = map(length, axes(A))
pinv(A::PInvQuasiMatrix) = first(A.applied.args)

@propagate_inbounds getindex(A::PInvQuasiMatrix{T}, k::Int, j::Int) where T =
    (A.pinv*[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

*(A::PInvQuasiMatrix, B::AbstractQuasiMatrix, C...) = apply(*,A.applied, B, C...)
*(A::PInvQuasiMatrix, B::MulQuasiArray, C...) = apply(*,A.applied, B.applied, C...)


####
# Matrix * Array
####

# the default is always Array

_materialize(M::QuasiArrayMulArray, _) = MulQuasiArray(M)
_materialize(M::ArrayMulQuasiArray, _) = MulQuasiArray(M)
_materialize(M::QuasiArrayMulQuasiArray, _) = MulQuasiArray(M)



# if multiplying two MulQuasiArrays simplifies the arguments, we materialize,
# otherwise we leave it as a lazy object
_mulquasi_join(As, M::MulQuasiArray, Cs) = MulQuasiArray(As..., M.applied.args..., Cs...)
_mulquasi_join(As, B, Cs) = *(As..., B, Cs...)


function _materialize(M::Mul2{<:Any,<:Any,<:MulQuasiArray,<:MulQuasiArray}, _)
    As, Bs = M.args
    _mul_join(reverse(tail(reverse(As))), last(As) * first(Bs), tail(Bs))
end


function _materialize(M::Mul2{<:Any,<:Any,<:MulQuasiArray,<:AbstractQuasiArray}, _)
    As, B = M.args
    rmaterialize(Mul(As.applied.args..., B))
end

function _materialize(M::Mul2{<:Any,<:Any,<:AbstractQuasiArray,<:MulQuasiArray}, _)
    A, Bs = M.args
    *(A, Bs.applied.args...)
end

# A MulQuasiArray can't be materialized further left-to-right, so we do right-to-left
function _materialize(M::Mul2{<:Any,<:Any,<:MulQuasiArray,<:AbstractArray}, _)
    As, B = M.args
    rmaterialize(Mul(As.applied.args..., B))
end

function _lmaterialize(A::MulQuasiArray, B, C...)
    As = A.applied.args
    flatten(_MulArray(reverse(tail(reverse(As)))..., _lmaterialize(last(As), B, C...)))
end



function _rmaterialize(Z::MulQuasiArray, Y, W...)
    Zs = Z.applied.args
    flatten(_MulArray(_rmaterialize(first(Zs), Y, W...), tail(Zs)...))
end
