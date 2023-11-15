
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
struct InvPlan{T, Facts, Dims} <: Plan{T}
    factorizations::Facts
    dims::Dims
end

InvPlan(fact, dims) = InvPlan{eltype(fact), typeof(fact), typeof(dims)}(fact, dims)

size(F::InvPlan) = size.(F.factorizations, 1)
size(F::InvPlan{<:Any,<:Any,Int}) = size(F.factorizations, 1)


function *(P::InvPlan{<:Any,<:Any,Int}, x::AbstractVector)
    @assert P.dims == 1
    P.factorizations \ x # Only a single factorization when dims isa Int
end

function *(P::InvPlan{<:Any,<:Any,Int}, X::AbstractMatrix)
    if P.dims == 1
        P.factorizations \ X  # Only a single factorization when dims isa Int
    else
        @assert P.dims == 2
        permutedims(P.factorizations \ permutedims(X))
    end
end

function *(P::InvPlan{<:Any,<:Any,Int}, X::AbstractArray{<:Any,3})
    Y = similar(X)
    if P.dims == 1
        for j in axes(X,3)
            Y[:,:,j] = P.factorizations \ X[:,:,j]
        end
    elseif P.dims == 2
        for k in axes(X,1)
            Y[k,:,:] = P.factorizations \ X[k,:,:]
        end
    else
        @assert P.dims == 3
        for k in axes(X,1), j in axes(X,2)
            Y[k,j,:] = P.factorizations \ X[k,j,:]
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
struct MulPlan{T, Fact, Dims} <: Plan{T}
    matrices::Fact
    dims::Dims
end

MulPlan(mats, dims) = MulPlan{eltype(mats), typeof(mats), typeof(dims)}(mats, dims)

function *(P::MulPlan{<:Any,<:Any,Int}, x::AbstractVector)
    @assert P.dims == 1
    P.matrices * x
end

function *(P::MulPlan{<:Any,<:Any,Int}, X::AbstractMatrix)
    if P.dims == 1
        P.matrices * X
    else
        @assert P.dims == 2
        permutedims(P.matrices * permutedims(X))
    end
end

function *(P::MulPlan{<:Any,<:Any,Int}, X::AbstractArray{<:Any,3})
    Y = similar(X)
    if P.dims == 1
        for j in axes(X,3)
            Y[:,:,j] = P.matrices * X[:,:,j]
        end
    elseif P.dims == 2
        for k in axes(X,1)
            Y[k,:,:] = P.matrices * X[k,:,:]
        end
    else
        @assert P.dims == 3
        for k in axes(X,1), j in axes(X,2)
            Y[k,j,:] = P.matrices * X[k,j,:]
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
inv(P::InvPlan) = MulPlan(P.factorizations, P.dims)