
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
struct InvPlan{T, Fact, Dims} <: Plan{T}
    factorization::Fact
    dims::Dims
end

InvPlan(fact, dims) = InvPlan{eltype(fact), typeof(fact), typeof(dims)}(fact, dims)

size(F::InvPlan, k...) = size(F.factorization, k...)


function *(P::InvPlan{<:Any,<:Any,Int}, x::AbstractVector)
    @assert P.dims == 1
    P.factorization \ x
end

function *(P::InvPlan{<:Any,<:Any,Int}, X::AbstractMatrix)
    if P.dims == 1
        P.factorization \ X
    else
        @assert P.dims == 2
        permutedims(P.factorization \ permutedims(X))
    end
end

function *(P::InvPlan{<:Any,<:Any,Int}, X::AbstractArray{<:Any,3})
    Y = similar(X)
    if P.dims == 1
        for j in axes(X,3)
            Y[:,:,j] = P.factorization \ X[:,:,j]
        end
    elseif P.dims == 2
        for k in axes(X,1)
            Y[k,:,:] = P.factorization \ X[k,:,:]
        end
    else
        @assert P.dims == 3
        for k in axes(X,1), j in axes(X,2)
            Y[k,j,:] = P.factorization \ X[k,j,:]
        end
    end
    Y
end

function *(P::InvPlan, X::AbstractArray)
    for d in P.dims
        X = InvPlan(P.factorization, d) * X
    end
    X
end


"""
    MulPlan(matrix, dims)

Takes a matrix and supports it applied to different dimensions.
"""
struct MulPlan{T, Fact, Dims} <: Plan{T}
    matrix::Fact
    dims::Dims
end

MulPlan(fact, dims) = MulPlan{eltype(fact), typeof(fact), typeof(dims)}(fact, dims)

function *(P::MulPlan{<:Any,<:Any,Int}, x::AbstractVector)
    @assert P.dims == 1
    P.matrix * x
end

function *(P::MulPlan{<:Any,<:Any,Int}, X::AbstractMatrix)
    if P.dims == 1
        P.matrix * X
    else
        @assert P.dims == 2
        permutedims(P.matrix * permutedims(X))
    end
end

function *(P::MulPlan{<:Any,<:Any,Int}, X::AbstractArray{<:Any,3})
    Y = similar(X)
    if P.dims == 1
        for j in axes(X,3)
            Y[:,:,j] = P.matrix * X[:,:,j]
        end
    elseif P.dims == 2
        for k in axes(X,1)
            Y[k,:,:] = P.matrix * X[k,:,:]
        end
    else
        @assert P.dims == 3
        for k in axes(X,1), j in axes(X,2)
            Y[k,j,:] = P.matrix * X[k,j,:]
        end
    end
    Y
end

function *(P::MulPlan, X::AbstractArray)
    for d in P.dims
        X = MulPlan(P.matrix, d) * X
    end
    X
end

*(A::AbstractMatrix, P::MulPlan) = MulPlan(A*P.matrix, P.dims)

inv(P::MulPlan) = InvPlan(factorize(P.matrix), P.dims)
inv(P::InvPlan) = MulPlan(P.factorization, P.dims)