

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
    elseif qt.args[1].args[1] == :(\)
        mat = qt.args[2]
        @assert qt.args[1].args[2].head == :(::)
        Aname,Atyp = qt.args[1].args[2].args
        Bname,Btyp = qt.args[1].args[3].args
        esc(quote
            ContinuumArrays.simplifiable(::typeof(\), A::$Atyp, B::$Btyp) = Val(true)
            Base.@propagate_inbounds function ContinuumArrays.ldiv($Aname::$Atyp, $Bname::$Btyp)
                @boundscheck ContinuumArrays.check_ldiv_axes($Aname, $Bname)
                $mat
            end
        end)
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
struct Derivative{T,D,Order} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
    order::Order
end

Derivative{T, D}(axis::Inclusion{<:Any,D}, order) where {T,D} = Derivative{T,D,typeof(order)}(axis, order)
Derivative{T, D}(axis::Inclusion{<:Any,D}) where {T,D} = Derivative{T,D,Nothing}(axis, nothing)
Derivative{T}(axis::Inclusion{<:Any,D}, order...) where {T,D} = Derivative{T,D}(axis, order...)
Derivative{T}(domain, order...) where T = Derivative{T}(Inclusion(domain), order...)

Derivative(ax::AbstractQuasiVector{T}, order...) where T = Derivative{eltype(ax)}(ax, order...)
Derivative(L::AbstractQuasiMatrix, order...) = Derivative(axes(L,1), order...)

show(io::IO, a::Derivative) = summary(io, a)
function summary(io::IO, D::Derivative{T,Dom,Nothing}) where {T,Dom}
    print(io, "Derivative(")
    summary(io, D.axis)
    print(io,")")
end

function summary(io::IO, D::Derivative)
    print(io, "Derivative(")
    summary(io, D.axis)
    print(io, ", ")
    print(io, D.order)
    print(io,")")
end

axes(D::Derivative) = (D.axis, D.axis)
==(a::Derivative, b::Derivative) = a.axis == b.axis && a.order == b.order
copy(D::Derivative) = D


@simplify function *(D::Derivative, B::AbstractQuasiVecOrMat)
    T = typeof(zero(eltype(D)) * zero(eltype(B)))
    if D.order isa Nothing
        diff(convert(AbstractQuasiArray{T}, B))
    else
        diff(convert(AbstractQuasiArray{T}, B), D.order)
    end
end



^(D::Derivative{T,Dom,Nothing}, k::Integer) where {T,Dom} = Derivative{T}(D.axis, k)
^(D::Derivative{T}, k::Integer) where T = Derivative{T}(D.axis, D.order .* k)


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

struct OperatorLayout <: AbstractLazyLayout end
MemoryLayout(::Type{<:Derivative}) = OperatorLayout()
# copy(M::Mul{OperatorLayout, <:ExpansionLayout}) = simplify(M)
# copy(M::Mul{OperatorLayout, <:AbstractLazyLayout}) = M.A * expand(M.B)


# Laplacian

struct Laplacian{T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Laplacian{T}(axis::Inclusion{<:Any,D}) where {T,D} = Laplacian{T,D}(axis)
# Laplacian{T}(domain) where T = Laplacian{T}(Inclusion(domain))
# Laplacian(axis) = Laplacian{eltype(axis)}(axis)

axes(D::Laplacian) = (D.axis, D.axis)
==(a::Laplacian, b::Laplacian) = a.axis == b.axis
copy(D::Laplacian) = Laplacian(copy(D.axis))

@simplify function *(D::Laplacian, B::AbstractQuasiVecOrMat)
    T = typeof(zero(eltype(D)) * zero(eltype(B)))
    laplacian(convert(AbstractQuasiArray{T}, B))
end



# Negative fractional Laplacian (-Δ)^α or equiv. abs(Δ)^α

struct AbsLaplacian{T,D,A} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
    order::A
end

AbsLaplacian{T}(axis::Inclusion{<:Any,D},α=1) where {T,D} = AbsLaplacian{T,D,typeof(α)}(axis,α)

axes(D:: AbsLaplacian) = (D.axis, D.axis)
==(a:: AbsLaplacian, b:: AbsLaplacian) = a.axis == b.axis && a.α == b.α
copy(D:: AbsLaplacian) = AbsLaplacian(copy(D.axis), D.α)

abs(Δ::Laplacian) = AbsLaplacian(axes(Δ,1))
-(Δ::Laplacian) = abs(Δ)

^(D::AbsLaplacian, k) = AbsLaplacian(D.axis, D.α*k)