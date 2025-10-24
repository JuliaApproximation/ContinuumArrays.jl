
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
struct InvPlan{T, Facts<:Tuple, Pln, Dims} <: Plan{T}
    factorizations::Facts
    plan::Pln
    dims::Dims
end

InvPlan(fact::Tuple, plan, dims) = InvPlan{mapreduce(eltype,promote_type,fact), typeof(fact), typeof(plan), typeof(dims)}(fact, plan, dims)
InvPlan(fact::Tuple, dims) = InvPlan(fact, nothing, dims)
InvPlan(fact, dims...) = InvPlan((fact,), dims...)

size(F::InvPlan) = size.(F.factorizations, 1)


"""
    MulPlan(matrix, [plan], dims)

Takes a matrix and supports it applied to different dimensions, after applying a plan.
"""
struct MulPlan{T, Fact<:Tuple, Pln, Dims} <: Plan{T}
    matrices::Fact
    plan::Pln
    dims::Dims
end

MulPlan(mats::Tuple, plan, dims) = MulPlan{mapreduce(eltype,promote_type,mats), typeof(mats), typeof(plan), typeof(dims)}(mats, plan, dims)
MulPlan(mats::Tuple, dims) = MulPlan(mats, nothing, dims)
MulPlan(mats::AbstractMatrix, dims...) = MulPlan((mats,), dims...)

_transformifnotnothing(::Nothing, x) = x
_transformifnotnothing(P, x) = P*x

for (Pln,op,fld) in ((:MulPlan, :*, :(:matrices)), (:InvPlan, :\, :(:factorizations)))
    @eval begin
        function *(P::$Pln{<:Any,<:Tuple,<:Any,Int}, x::AbstractVector)
            @assert P.dims == 1
            $op(only(getfield(P, $fld)), _transformifnotnothing(P.plan, x)) # Only a single factorization when dims isa Int
        end
        
        function *(P::$Pln{<:Any,<:Tuple,<:Any,Int}, X::AbstractMatrix)
            if P.dims == 1
                $op(only(getfield(P, $fld)), X)  # Only a single factorization when dims isa Int
            else
                @assert P.dims == 2
                permutedims($op(only(getfield(P, $fld)), permutedims(_transformifnotnothing(P.plan, X))))
            end
        end
        
        function *(P::$Pln{<:Any,<:Tuple,<:Any,Int}, Xin::AbstractArray{<:Any,3})
            X = _transformifnotnothing(P.plan, Xin)
            Y = similar(X)
            if P.dims == 1
                for j in axes(X,3)
                    Y[:,:,j] = $op(only(getfield(P, $fld)), X[:,:,j])
                end
            elseif P.dims == 2
                for k in axes(X,1)
                    Y[k,:,:] = $op(only(getfield(P, $fld)), X[k,:,:])
                end
            else
                @assert P.dims == 3
                for k in axes(X,1), j in axes(X,2)
                    Y[k,j,:] = $op(only(getfield(P, $fld)), X[k,j,:])
                end
            end
            Y
        end
        
        function *(P::$Pln{<:Any,<:Tuple,<:Any,Int}, Xin::AbstractArray{<:Any,4})
            X = _transformifnotnothing(P.plan, Xin)
            Y = similar(X)
            if P.dims == 1
                for j in axes(X,3), l in axes(X,4)
                    Y[:,:,j,l] = $op(only(getfield(P, $fld)), X[:,:,j,l])
                end
            elseif P.dims == 2
                for k in axes(X,1), l in axes(X,4)
                    Y[k,:,:,l] = $op(only(getfield(P, $fld)), X[k,:,:,l])
                end
            elseif P.dims == 3
                for k in axes(X,1), j in axes(X,2)
                    Y[k,j,:,:] = $op(only(getfield(P, $fld)), X[k,j,:,:])
                end
            elseif P.dims == 4
                for k in axes(X,1), j in axes(X,2), l in axes(X,3)
                    Y[k,j,l,:] = $op(only(getfield(P, $fld)), X[k,j,l,:])
                end
            end
            Y
        end
        
        
        
        *(P::$Pln{<:Any,<:Tuple,<:Any,Int}, X::AbstractArray) = error("Overload")
        
        function *(P::$Pln, Xin::AbstractArray)
            X = _transformifnotnothing(P.plan, Xin)
            for (fac,dim) in zip(getfield(P, $fld), P.dims)
                X = $Pln(fac, dim) * X
            end
            X
        end
    end
end

*(A::AbstractMatrix, P::MulPlan) = MulPlan(Ref(A) .* P.matrices, P.plan, P.dims)

inv(P::MulPlan{<:Any,<:Any,Nothing}) = InvPlan(map(factorize,P.matrices), P.dims)
inv(P::InvPlan{<:Any,<:Any,Nothing}) = MulPlan(convert.(Matrix,P.factorizations), P.dims)