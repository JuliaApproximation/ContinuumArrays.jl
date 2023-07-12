

function removesubtype(typ)
    if typ isa Expr && typ.head == :(<:) 
        typ.args[1]
    else
        typ
    end
end

function adjtype(Atyp) 
    if Atyp isa Expr && Atyp.args[1] == :QuasiAdjoint 
        removesubtype(Atyp.args[3])
    else
        :(QuasiAdjoint{<:Any,<:$Atyp}) 
    end
end

macro simplify(qt)
    qt.head == :function || qt.head == :(=) || error("Must start with a function")
    @assert qt.args[1].head == :call
    if qt.args[1].args[1] == :(*)
        mat = qt.args[2]
        @assert qt.args[1].args[2].head == :(::)
        Aname,Atyp = qt.args[1].args[2].args
        Aadj = adjtype(Atyp)
        Bname,Btyp = qt.args[1].args[3].args
        Badj = adjtype(Btyp)
        if length(qt.args[1].args) == 3
            ret = quote
                ContinuumArrays.simplifiable(::typeof(*), A::$Atyp, B::$Btyp) = Val(true)
                Base.@propagate_inbounds function ContinuumArrays.mul($Aname::$Atyp, $Bname::$Btyp)
                    @boundscheck ContinuumArrays.check_mul_axes($Aname, $Bname)
                    $mat
                end
            end
            if Aadj ≠ Btyp 
                @assert Atyp ≠ Badj
                ret = quote
                    $ret
                    ContinuumArrays.simplifiable(::typeof(*), ::$Badj, A::$Aadj) = Val(true)
                    Base.@propagate_inbounds ContinuumArrays.mul(Bc::$Badj, Ac::$Aadj) = ContinuumArrays.mul(Ac', Bc')'
                end
            end
            
            esc(ret)
        else
            @assert length(qt.args[1].args) == 4
            Cname,Ctyp = qt.args[1].args[4].args
            esc(quote
                ContinuumArrays.simplifiable(::typeof(*), ::$Atyp, ::$Btyp, ::$Ctyp) = Val(true)
                function ContinuumArrays._simplify(::typeof(*), $Aname::$Atyp, $Bname::$Btyp, $Cname::$Ctyp)
                    $mat
                end
                Base.copy(M::ContinuumArrays.QMul3{<:$Atyp,<:$Btyp,<:$Ctyp}) = ContinuumArrays.simplify(M)
            end)
        end
    end
end



struct DiracDelta{T,A} <: LazyQuasiVector{T}
    x::T
    axis::A
    DiracDelta{T,A}(x, axis) where {T,A} = new{T,A}(x,axis)
end

DiracDelta{T}(x, axis::A) where {T,A<:AbstractQuasiVector} = DiracDelta{T,A}(x, axis)
DiracDelta{T}(x, domain) where T = DiracDelta{T}(x, Inclusion(domain))
DiracDelta(x, domain) = DiracDelta{float(eltype(x))}(x, Inclusion(domain))
DiracDelta(x) = DiracDelta(x, Inclusion(x))
DiracDelta{T}() where T = DiracDelta(zero(T))
DiracDelta() = DiracDelta{Float64}()


axes(δ::DiracDelta) = (δ.axis,)
IndexStyle(::Type{<:DiracDelta}) = IndexLinear()

==(a::DiracDelta, b::DiracDelta) = a.axis == b.axis && a.x == b.x

function getindex(δ::DiracDelta{T}, x::Number) where T
    x ∈ δ.axis || throw(BoundsError())
    convert(T, x == δ.x ? inv(zero(T)) : zero(T))::T
end


@simplify *(A::QuasiAdjoint{<:Any,<:DiracDelta}, B::AbstractQuasiMatrix) = B[parent(A).x,:]
dot(δ::DiracDelta, B::AbstractQuasiVector) = B[δ.x]
@simplify *(A::QuasiAdjoint{<:Any,<:DiracDelta}, B::AbstractQuasiVector) = dot(parent(A),B)

show(io::IO, δ::DiracDelta) = print(io, "δ at $(δ.x) over $(axes(δ,1))")
show(io::IO, ::MIME"text/plain", δ::DiracDelta) = show(io, δ)

#########
# Differentiation
#########

"""
Derivative(axis)

represents the differentiation (or finite-differences) operator on the
specified axis.
"""
struct Derivative{T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Derivative{T}(axis::Inclusion{<:Any,D}) where {T,D} = Derivative{T,D}(axis)
Derivative{T}(domain) where T = Derivative{T}(Inclusion(domain))

Derivative(L::AbstractQuasiMatrix) = Derivative(axes(L,1))

show(io::IO, a::Derivative) = summary(io, a)
function summary(io::IO, D::Derivative)
    print(io, "Derivative(")
    summary(io,D.axis)
    print(io,")")
end

axes(D::Derivative) = (D.axis, D.axis)
==(a::Derivative, b::Derivative) = a.axis == b.axis
copy(D::Derivative) = Derivative(copy(D.axis))

@simplify function *(D::Derivative, B::AbstractQuasiMatrix)
    T = typeof(zero(eltype(D)) * zero(eltype(B)))
    diff(convert(AbstractQuasiMatrix{T}, B); dims=1)
end

@simplify function *(D::Derivative, B::AbstractQuasiVector)
    T = typeof(zero(eltype(D)) * zero(eltype(B)))
    diff(convert(AbstractQuasiVector{T}, B))
end




^(D::Derivative, k::Integer) = ApplyQuasiArray(^, D, k)


function view(D::Derivative, kr::Inclusion, jr::Inclusion)
    @boundscheck axes(D,1) == kr == jr || throw(BoundsError(D,(kr,jr)))
    D
end

# struct Multiplication{T,F,A} <: AbstractQuasiMatrix{T}
#     f::F
#     axis::A
# end


const Identity{T,D} = QuasiDiagonal{T,Inclusion{T,D}}

Identity(d::Inclusion) = QuasiDiagonal(d)

diff(x::Inclusion; dims::Integer=1) = ones(eltype(x), x)
diff(c::AbstractQuasiFill{<:Any,1}; dims::Integer=1) =  zeros(eltype(c), axes(c,1))

struct OperatorLayout <: AbstractLazyLayout end
MemoryLayout(::Type{<:Derivative}) = OperatorLayout()
# copy(M::Mul{OperatorLayout, <:ExpansionLayout}) = simplify(M)
# copy(M::Mul{OperatorLayout, <:AbstractLazyLayout}) = M.A * expand(M.B)
