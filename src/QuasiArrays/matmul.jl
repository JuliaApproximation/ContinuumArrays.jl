
@inline MemoryLayout(A::AbstractQuasiArray{T}) where T = UnknownLayout()

const QuasiArrayMulArray{TV, styleA, styleB, p, q, T, V} =
    Mul{TV, styleA, styleB, <:AbstractQuasiArray{T,p}, <:AbstractArray{V,q}}

####
# Matrix * Vector
####
let (p,q) = (2,1)
    global const QuasiMatMulVec{TV, styleA, styleB, T, V} = QuasiArrayMulArray{TV, styleA, styleB, p, q, T, V}
end

axes(M::QuasiMatMulVec) = (axes(M.A,1),)

function getindex(M::QuasiMatMulVec{T}, k::Real) where T
    ret = zero(T)
    for j = 1:size(M.A,2)
        ret += M.A[k,j] * M.B[j]
    end
    ret
end

*(A::AbstractQuasiArray, B::AbstractQuasiArray) = materialize(Mul(A,B))
*(A::AbstractQuasiArray, B::AbstractArray) = materialize(Mul(A,B))
*(A::AbstractArray, B::AbstractQuasiArray) = materialize(Mul(A,B))
inv(A::AbstractQuasiArray) = materialize(Inv(A))
*(A::Inv{<:Any,<:Any,<:AbstractQuasiArray}, B::AbstractQuasiArray) = materialize(Mul(A,B))
