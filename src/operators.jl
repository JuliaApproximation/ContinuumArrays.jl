

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
struct Derivative{T,D<:Inclusion,Order} <: LazyQuasiMatrix{T}
    axis::D
    order::Order
end

"""
Laplacian(axis)

represents the laplacian operator `Δ` on the
specified axis.
"""
struct Laplacian{T,D<:Inclusion,Order} <: LazyQuasiMatrix{T}
    axis::D
    order::Order
end

"""
AbsLaplacian(axis)

represents the positive-definite/negative/absolute-value laplacian operator `|Δ| ≡ -Δ` on the
specified axis.
"""
struct AbsLaplacian{T,D<:Inclusion,Order} <: LazyQuasiMatrix{T}
    axis::D
    order::Order
end

_operatororder(D) = something(D.order, 1)

for (Op, op) in ((:Derivative, :diff), (:Laplacian, :laplacian), (:AbsLaplacian, :abslaplacian))
    @eval begin
        $Op{T, D}(axis::D, order) where {T,D<:Inclusion} = $Op{T,D,typeof(order)}(axis, order)
        $Op{T, D}(axis::D) where {T,D<:Inclusion} = $Op{T,D,Nothing}(axis, nothing)
        $Op{T}(axis::D, order...) where {T,D<:Inclusion} = $Op{T,D}(axis, order...)
        $Op{T}(domain, order...) where T = $Op{T}(Inclusion(domain), order...)

        $Op(ax::AbstractQuasiVector{T}, order...) where T = $Op{eltype(eltype(ax))}(ax, order...)
        $Op(L::AbstractQuasiMatrix, order...) = $Op(axes(L,1), order...)

        show(io::IO, a::$Op) = summary(io, a)
        function summary(io::IO, D::$Op{<:Any,<:Inclusion,Nothing})
            print(io, "$($Op)(")
            summary(io, D.axis)
            print(io,")")
        end

        function summary(io::IO, D::$Op)
            print(io, "$($Op)(")
            summary(io, D.axis)
            print(io, ", ")
            print(io, D.order)
            print(io,")")
        end

        axes(D::$Op) = (D.axis, D.axis)
        ==(a::$Op, b::$Op) = a.axis == b.axis && _operatororder(a) == _operatororder(b)
        copy(D::$Op) = D


        @simplify function *(D::$Op, B::AbstractQuasiVecOrMat)
            T = typeof(zero(eltype(D)) * zero(eltype(B)))
            if D.order isa Nothing
                $op(convert(AbstractQuasiArray{T}, B))
            else
                $op(convert(AbstractQuasiArray{T}, B), D.order)
            end
        end

        ^(D::$Op{T}, k::Integer) where T = $Op{T}(D.axis, _operatororder(D) .* k)

        @simplify *(D1::$Op, D2::$Op) = $Op{promote_type(eltype(D1),eltype(D2))}(D1.axis, _operatororder(D1)+_operatororder(D2))

        function view(D::$Op, kr::Inclusion, jr::Inclusion)
            @boundscheck axes(D,1) == kr == jr || throw(BoundsError(D,(kr,jr)))
            D
        end
    end
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

abs(Δ::Laplacian{T}) where T = AbsLaplacian{T}(axes(Δ,1), Δ.order)
-(Δ::Laplacian{<:Any,<:Any,Nothing}) = abs(Δ)
-(Δ::AbsLaplacian{T,<:Any,Nothing}) where T = Laplacian{T}(Δ.axis)

^(Δ::AbsLaplacian{T}, k::Real) where T = AbsLaplacian{T}(Δ.axis, _operatororder(Δ)*k)