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
