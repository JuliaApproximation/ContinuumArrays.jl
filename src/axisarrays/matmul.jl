
@inline MemoryLayout(A::AbstractAxisArray{T}) where T = UnknownLayout()

const AxisArrayMulArray{TV, styleA, styleB, p, q, T, V} =
    Mul{TV, styleA, styleB, <:AbstractAxisArray{T,p}, <:AbstractArray{V,q}}

####
# Matrix * Vector
####
let (p,q) = (2,1)
    global const AxisMatMulVec{TV, styleA, styleB, T, V} = AxisArrayMulArray{TV, styleA, styleB, p, q, T, V}
end

axes(M::AxisMatMulVec) = (axes(M.A,1),)

function getindex(M::AxisMatMulVec{T}, k::Real) where T
    ret = zero(T)
    for j = 1:size(M.A,2)
        ret += M.A[k,j] * M.B[j]
    end
    ret
end

materialize(M) = M
*(A::AbstractAxisArray, B::AbstractAxisArray) = materialize(Mul(A,B))
*(A::AbstractAxisArray, B::AbstractArray) = materialize(Mul(A,B))
*(A::AbstractArray, B::AbstractAxisArray) = materialize(Mul(A,B))
inv(A::AbstractAxisArray) = materialize(Inv(A))
*(A::Inv{<:Any,<:Any,<:AbstractAxisArray}, B::AbstractAxisArray) = materialize(Mul(A,B))
