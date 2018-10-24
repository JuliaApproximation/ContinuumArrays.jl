## Option 1


abstract type BasisFunction <: Function end
struct LegendreP <: BasisFunction
    n::Int
end
Base.transpose(x::Legendre) = x



B = transpose(LegendreP.(1:∞))
v = Vcat([1,2,3], Zeros(∞))

f = Mul(B, v)



## Option 2
using LazyArrays, InfiniteArrays, LinearAlgebra
import Base: size, getindex, show, +, *, -, convert, copyto!
import LinearAlgebra: adjoint, SymTridiagonal
import InfiniteArrays: Infinity

abstract type QuasiMatrix <: AbstractMatrix{Any} end

const Vec{QM<:QuasiMatrix} = Mul{<:Any,<:Any,<:Any,QM}

Base.IndexStyle(::QuasiMatrix) = Base.IndexLinear()

for op in (:+, :-)
    @eval $op(f::Vec{BASIS}, g::Vec{BASIS}) where BASIS <: QuasiMatrix =
        Mul(f.A, $op(f.B , g.B))
end

*(c::Number, f::Vec) = Mul(f.A, c*f.B)

adjoint(f::Vec) = Adjoint(f)

show(io::IO, f::Adjoint{<:Any,<:Vec}) = print(io, "Bra")
show(io::IO, ::MIME"text/plain", f::Adjoint{<:Any,<:Vec}) = print(io, "Bra $(typeof(f))")

show(io::IO, B::QuasiMatrix) = print(io, string(typeof(B)))
show(io::IO, ::MIME"text/plain", B::QuasiMatrix) = print(io, string(typeof(B)))

pad(v, n::Infinity) = Vcat(v, Zeros(n-length(v)))
pad(v, n) = vcat(v, Zeros(n-length(v)))

getindex(B::QuasiMatrix, k) = Mul(B, pad([Zeros(k-1); 1], size(B,2)))

struct DiracDelta
    x::Float64
end

*(δ::DiracDelta, f::Vec) = (δ*f.A) * f.B



struct Legendre <: QuasiMatrix  end

size(::Legendre) = (1,∞)


struct LinearSpline <: QuasiMatrix
    points::Vector{Float64}
end

size(B::LinearSpline) = (1,length(B.points))


function convert(::Type{SymTridiagonal}, AB::Mul{<:Any,<:Any,<:Any, <:Adjoint{<:Any,LinearSpline},<:LinearSpline})
    Ac,B = AB.A, AB.B
    A = parent(Ac)
    @assert A.points == B.points
    x = A.points
    SymTridiagonal(x, x/2) # TODO fix
end

SymTridiagonal(AB::Mul) = convert(SymTridiagonal, AB)


copyto!(C::AbstractMatrix, AB::Mul{<:Any,<:Any,<:Any, <:Adjoint{<:Any,LinearSpline},<:LinearSpline}) =
    copyto!(C, SymTridiagonal(AB))

*(Ac::Adjoint{<:Any,LinearSpline}, B::LinearSpline) = SymTridiagonal(Mul(Ac, B))

function *(δ::DiracDelta, B::LinearSpline)
    x = δ.x
    @assert B.points[1] ≤ B.points[2]
    [(B.points[2]-B.points[1])*(x-B.points[1]);Zeros(size(B,2)-1)]'
end

(f::Vec{LinearSpline})(x) = DiracDelta(x) * f


SymTridiagonal(Mul(B', B))


C = Array{Float64}(undef, 3, 3)

B = LinearSpline([1,2,3])
size(B)


copyto!(C,  Mul(B', B))


C .=  Mul(B', B)


size(Mul(B', B))


A = randn(5,5)
B = randn(5,5)
C = similar(A)

C .= 2.0 .* Mul(A,B) .+ 3.0 .* C
δ = DiracDelta(0)

δ*f
f(0.5)

f = Mul(B, [1,2,3])

1B[1] + 2B[2] + 3B[3] - f


DiracDelta(2) * B

DiracDelta(2) * f


B = Legendre()
Bt = B'; (B[1]')

B' * B

Eye(∞)


B = LinearSpline([1,2,3])

B'*B

typeof(B')
typeof(B)


factorize(B)


## Option 3 ContinuumArrays.jl

using LinearAlgebra, IntervalSets, FillArrays, LazyArrays
import Base: size, getindex, show, +, *, -, convert, copyto!, length, axes, parent, eltype
import LinearAlgebra: adjoint, SymTridiagonal
import InfiniteArrays: Infinity
import LazyArrays: MemoryLayout

abstract type ContinuumArray{T, N} end
const ContinuumVector{T} = ContinuumArray{T, 1}
const ContinuumMatrix{T} = ContinuumArray{T, 2}

eltype(::Type{<:ContinuumArray{T}}) where T = T
eltype(::ContinuumArray{T}) where T = T

struct ℵ₀ <: Number end
_length(::AbstractInterval) = ℵ₀
_length(d) = length(d)

size(A::ContinuumArray) = _length.(axes(A))
axes(A::ContinuumArray, j::Int) = axes(A)[j]
size(A::ContinuumArray, j::Int) = size(A)[j]

struct ContinuumLayout <: MemoryLayout end
MemoryLayout(::ContinuumArray) = ContinuumLayout()

getindex(B::ContinuumMatrix, K::AbstractVector, j::Real) =
    [B[k, j] for k in K]

getindex(B::ContinuumMatrix, k::Real, J::AbstractVector) =
    [B[k, j] for j in J]

getindex(B::ContinuumMatrix, K::AbstractVector, J::AbstractVector) =
    [B[k, j] for k in K, j in J]

getindex(B::ContinuumMatrix, ::Colon, ::Colon) = copy(B)
getindex(B::ContinuumMatrix, ::Colon, J) = B[:, J]
getindex(B::ContinuumMatrix, K, ::Colon) = B[K, axes(B,2)]


# use lazy multiplication
materialize(M::Mul) = M
*(A::ContinuumArray, B::ContinuumArray) = materialize(Mul(A,B))

struct CAdjoint{T,PAR} <: ContinuumMatrix{T}
    parent::PAR
end

CAdjoint(A::ContinuumArray{T}) where T = CAdjoint{T, typeof(A)}(A)

parent(A::CAdjoint) = A.parent

axes(A::CAdjoint) = reverse(axes(parent(A)))
adjoint(A::ContinuumArray) = CAdjoint(A)
getindex(A::CAdjoint, k::Real, j::Real) = adjoint(parent(A)[j,k])



struct LinearSpline{T} <: ContinuumMatrix{T}
    points::Vector{T}
end

LinearSpline(p::Vector{T}) where T = LinearSpline{float(T)}(p)

axes(B::LinearSpline) = (first(B.points)..last(B.points), Base.OneTo(length(B.points)))

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

getindex(B::LinearSpline, ::Colon, k::Int) = Mul(B, [Zeros{Int}(k-1); 1; Zeros{Int}(size(B,2)-k)])

function convert(::Type{SymTridiagonal}, AB::Mul{<:Any,<:Any,<:Any,<:CAdjoint{<:Any,<:LinearSpline},<:LinearSpline})
    Ac,B = AB.A, AB.B
    A = parent(Ac)
    @assert A.points == B.points
    x = A.points
    SymTridiagonal(x, x/2) # TODO fix
end

materialize(M::Mul{<:Any,<:Any,<:Any,<:CAdjoint{<:Any,<:LinearSpline},<:LinearSpline}) =
    convert(SymTridiagonal, M)

## tests

B = LinearSpline([1,2,3])

using Plots
x = range(1, stop=3, length=1000)
plot(B[x,:])


@test B'B isa SymTridiagonal
