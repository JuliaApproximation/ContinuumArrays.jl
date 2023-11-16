
struct TransformFactorization{T,Grid,Plan} <: Factorization{T}
    grid::Grid
    plan::Plan
end

TransformFactorization{T}(grid, plan) where T = TransformFactorization{T,typeof(grid),typeof(plan)}(grid, plan)

"""
    TransformFactorization(grid, plan)

associates a planned transform with a grid. That is, if `F` is a `TransformFactorization`, then
`F \\ f` is equivalent to `F.plan * f[F.grid]`.
"""
TransformFactorization(grid, plan) = TransformFactorization{promote_type(eltype(eltype(grid)),eltype(plan))}(grid, plan)



grid(T::TransformFactorization) = T.grid
function size(T::TransformFactorization, k)
    @assert k == 2 # TODO: make consistent
    size(T.plan,1)
end


\(a::TransformFactorization, b::AbstractQuasiVector) = a.plan * convert(Array, b[a.grid])
\(a::TransformFactorization, b::AbstractQuasiMatrix) = a.plan * convert(Array, b[a.grid,:])

"""
    InvPlan(factorization, dims)

Takes a factorization and supports it applied to different dimensions.
"""
struct InvPlan{T, Facts<:Tuple, Dims} <: Plan{T}
    factorizations::Facts
    dims::Dims
end

InvPlan(fact::Tuple, dims) = InvPlan{eltype(fact), typeof(fact), typeof(dims)}(fact, dims)
InvPlan(fact, dims) = InvPlan((fact,), dims)

size(F::InvPlan) = size.(F.factorizations, 1)


function *(P::InvPlan{<:Any,<:Tuple,Int}, x::AbstractVector)
    @assert P.dims == 1
    only(P.factorizations) \ x # Only a single factorization when dims isa Int
end

function *(P::InvPlan{<:Any,<:Tuple,Int}, X::AbstractMatrix)
    if P.dims == 1
        only(P.factorizations) \ X  # Only a single factorization when dims isa Int
    else
        @assert P.dims == 2
        permutedims(only(P.factorizations) \ permutedims(X))
    end
end

function *(P::InvPlan{<:Any,<:Tuple,Int}, X::AbstractArray{<:Any,3})
    Y = similar(X)
    if P.dims == 1
        for j in axes(X,3)
            Y[:,:,j] = only(P.factorizations) \ X[:,:,j]
        end
    elseif P.dims == 2
        for k in axes(X,1)
            Y[k,:,:] = only(P.factorizations) \ X[k,:,:]
        end
    else
        @assert P.dims == 3
        for k in axes(X,1), j in axes(X,2)
            Y[k,j,:] = only(P.factorizations) \ X[k,j,:]
        end
    end
    Y
end

function *(P::InvPlan, X::AbstractArray)
    for d in P.dims
        X = InvPlan(P.factorizations[d], d) * X
    end
    X
end


"""
    MulPlan(matrix, dims)

Takes a matrix and supports it applied to different dimensions.
"""
struct MulPlan{T, Fact<:Tuple, Dims} <: Plan{T}
    matrices::Fact
    dims::Dims
end

MulPlan(mats::Tuple, dims) = MulPlan{eltype(mats), typeof(mats), typeof(dims)}(mats, dims)
MulPlan(mats::AbstractMatrix, dims) = MulPlan((mats,), dims)

function *(P::MulPlan{<:Any,<:Tuple,Int}, x::AbstractVector)
    @assert P.dims == 1
    only(P.matrices) * x
end

function *(P::MulPlan{<:Any,<:Tuple,Int}, X::AbstractMatrix)
    if P.dims == 1
        only(P.matrices) * X
    else
        @assert P.dims == 2
        permutedims(only(P.matrices) * permutedims(X))
    end
end

function *(P::MulPlan{<:Any,<:Tuple,Int}, X::AbstractArray{<:Any,3})
    Y = similar(X)
    if P.dims == 1
        for j in axes(X,3)
            Y[:,:,j] = only(P.matrices) * X[:,:,j]
        end
    elseif P.dims == 2
        for k in axes(X,1)
            Y[k,:,:] = only(P.matrices) * X[k,:,:]
        end
    else
        @assert P.dims == 3
        for k in axes(X,1), j in axes(X,2)
            Y[k,j,:] = only(P.matrices) * X[k,j,:]
        end
    end
    Y
end

function *(P::MulPlan, X::AbstractArray)
    for d in P.dims
        X = MulPlan(P.matrices[d], d) * X
    end
    X
end

*(A::AbstractMatrix, P::MulPlan) = MulPlan(Ref(A) .* P.matrices, P.dims)

inv(P::MulPlan) = InvPlan(map(factorize,P.matrices), P.dims)
inv(P::InvPlan) = MulPlan(convert.(Matrix,P.factorizations), P.dims)