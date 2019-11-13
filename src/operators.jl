

struct SimplifyStyle <: AbstractQuasiArrayApplyStyle end

copy(A::Applied{SimplifyStyle}) = error("Override copy(::$(typeof(A)))")

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
                LazyArrays.ApplyStyle(::typeof(*), ::Type{<:$Atyp}, ::Type{<:$Btyp}) = ContinuumArrays.SimplifyStyle()
                function Base.copy(M::ContinuumArrays.QMul2{<:$Atyp,<:$Btyp})
                    $Aname,$Bname = M.args
                    $mat
                end
            end
            if Aadj ≠ Btyp 
                @assert Atyp ≠ Badj
                ret = quote
                    $ret
                    LazyArrays.ApplyStyle(::typeof(*), ::Type{<:$Aadj}, ::Type{<:$Badj}) = ContinuumArrays.SimplifyStyle()
                    function Base.copy(M::ContinuumArrays.QMul2{<:$Badj,<:$Aadj})
                        Bc,Ac = M.args
                        apply(*,Ac',Bc')'
                    end
                end
            end
            
            esc(ret)
        else
            @assert length(qt.args[1].args) == 4
            Cname,Ctyp = qt.args[1].args[4].args
            esc(quote
            LazyArrays.ApplyStyle(::typeof(*), ::Type{<:$Atyp}, ::Type{<:$Btyp}, ::Type{<:$Ctyp}) = ContinuumArrays.SimplifyStyle()
            function Base.copy(M::ContinuumArrays.QMul3{<:$Atyp,<:$Btyp,<:$Ctyp})
                $Aname,$Bname,$Cname = M.args
                $mat
            end
         end)
    end
    else # A\
        mat = qt.args[2]  
        @assert qt.args[1].args[1] == :(\)
        @assert qt.args[1].args[2].head == :(::)
        Aname,Atyp = qt.args[1].args[2].args
        if qt.args[1].args[3].head == :(::) # A \ B
            Bname,Btyp = qt.args[1].args[3].args
            esc(quote
                LazyArrays.ApplyStyle(::typeof(\), ::Type{<:$Atyp}, ::Type{<:$Btyp}) = ContinuumArrays.SimplifyStyle()
                function Base.copy(M::Applied{ContinuumArrays.SimplifyStyle,typeof(\),<:Tuple{<:$Atyp,<:$Btyp}})
                    $Aname,$Bname = M.args
                    $mat
                end
            end)
        else # A \ (B*C)
            @assert qt.args[1].args[3].head == :call
            @assert qt.args[1].args[3].args[1] == :(*)
            @assert qt.args[1].args[3].args[2].head == :(::)
            Bname,Btyp = qt.args[1].args[3].args[2].args
            @assert qt.args[1].args[3].args[3].head == :(::)
            Cname,Ctyp = qt.args[1].args[3].args[3].args   
            if length(qt.args[1].args[3].args) == 3
                esc(quote
                    LazyArrays.ApplyStyle(::typeof(\), ::Type{<:$Atyp}, ::Type{<:ContinuumArrays.QMul2{<:$Btyp,<:$Ctyp}}) = ContinuumArrays.SimplifyStyle()
                    function Base.copy(M::Applied{ContinuumArrays.SimplifyStyle,typeof(\),<:Tuple{<:$Atyp,<:ContinuumArrays.QMul2{<:$Btyp,<:$Ctyp}}})
                        $Aname,BC = M.args
                        $Bname,$Cname = BC.args
                        $mat
                    end
                end)
            else
                @assert length(qt.args[1].args[3].args) == 4
                Dname,Dtyp = qt.args[1].args[3].args[4].args   
                esc(quote
                    ApplyStyle(::typeof(\),::Type{<:$Atyp}, ::Type{<:ContinuumArrays.QMul3{<:$Btyp,<:$Ctyp,<:$Dtyp}}) = ContinuumArrays.SimplifyStyle()
                    function Base.copy(M::Applied{ContinuumArrays.SimplifyStyle,typeof(\),<:Tuple{<:$Atyp,<:ContinuumArrays.QMul3{<:$Btyp,<:$Ctyp,<:$Dtyp}}})
                        $Aname,BC = M.args
                        $Bname,$Cname,$Dname = BC.args
                        $mat
                    end
                end)
            end
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
DiracDelta(x, domain) = DiracDelta{eltype(x)}(x, Inclusion(domain))
DiracDelta(axis::AbstractQuasiVector) = DiracDelta(zero(float(eltype(axis))), axis)
DiracDelta(domain) = DiracDelta(Inclusion(domain))

axes(δ::DiracDelta) = (δ.axis,)
IndexStyle(::Type{<:DiracDelta}) = IndexLinear()

==(a::DiracDelta, b::DiracDelta) = a.axis == b.axis && a.x == b.x

function getindex(δ::DiracDelta{T}, x::Number) where T
    x ∈ δ.axis || throw(BoundsError())
    x == δ.x ? inv(zero(T)) : zero(T)
end


@simplify *(A::QuasiAdjoint{<:Any,<:DiracDelta}, B::AbstractQuasiVector) = B[parent(A).x]
@simplify *(A::QuasiAdjoint{<:Any,<:DiracDelta}, B::AbstractQuasiMatrix) = B[parent(A).x,:]

#########
# Derivative
#########


struct Derivative{T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Derivative{T}(axis::A) where {T,A<:Inclusion} = Derivative{T,A}(axis)
Derivative{T}(domain) where T = Derivative{T}(Inclusion(domain))
Derivative(axis) = Derivative{Float64}(axis)

axes(D::Derivative) = (D.axis, D.axis)
==(a::Derivative, b::Derivative) = a.axis == b.axis
copy(D::Derivative) = Derivative(copy(D.axis))



# struct Multiplication{T,F,A} <: AbstractQuasiMatrix{T}
#     f::F
#     axis::A
# end


const Identity{T,D} = QuasiDiagonal{T,Inclusion{T,D}}

Identity(d::Inclusion) = QuasiDiagonal(d)
