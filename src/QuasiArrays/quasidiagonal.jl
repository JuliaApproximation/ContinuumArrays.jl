# This file is a part of Julia. License is MIT: https://julialang.org/license

## QuasiDiagonal matrices

struct QuasiDiagonal{T,V<:AbstractQuasiVector{T}} <: AbstractQuasiMatrix{T}
    diag::V

    function QuasiDiagonal{T,V}(diag) where {T,V<:AbstractQuasiVector{T}}
        new{T,V}(diag)
    end
end
QuasiDiagonal(v::AbstractQuasiVector{T}) where {T} = QuasiDiagonal{T,typeof(v)}(v)
QuasiDiagonal{T}(v::AbstractQuasiVector) where {T} = QuasiDiagonal(convert(AbstractQuasiVector{T}, v)::AbstractQuasiVector{T})


"""
    QuasiDiagonal(A::AbstractQuasiMatrix)

Construct a matrix from the diagonal of `A`.

# Examples
```jldoctest
julia> A = [1 2 3; 4 5 6; 7 8 9]
3×3 Array{Int64,2}:
 1  2  3
 4  5  6
 7  8  9

julia> QuasiDiagonal(A)
3×3 QuasiDiagonal{Int64,Array{Int64,1}}:
 1  ⋅  ⋅
 ⋅  5  ⋅
 ⋅  ⋅  9
```
"""
QuasiDiagonal(A::AbstractQuasiMatrix) = QuasiDiagonal(diag(A))

Diagonal(A::AbstractQuasiArray) = QuasiDiagonal(A)


"""
    QuasiDiagonal(V::AbstractQuasiVector)

Construct a matrix with `V` as its diagonal.

# Examples
```jldoctest
julia> V = [1, 2]
2-element Array{Int64,1}:
 1
 2

julia> QuasiDiagonal(V)
2×2 QuasiDiagonal{Int64,Array{Int64,1}}:
 1  ⋅
 ⋅  2
```
"""
QuasiDiagonal(V::AbstractQuasiVector)

QuasiDiagonal(D::QuasiDiagonal) = D
QuasiDiagonal{T}(D::QuasiDiagonal{T}) where {T} = D
QuasiDiagonal{T}(D::QuasiDiagonal) where {T} = QuasiDiagonal{T}(D.diag)

AbstractQuasiMatrix{T}(D::QuasiDiagonal) where {T} = QuasiDiagonal{T}(D)
Matrix(D::QuasiDiagonal) = diagm(0 => D.diag)
Array(D::QuasiDiagonal) = Matrix(D)

# For D<:QuasiDiagonal, similar(D[, neweltype]) should yield a QuasiDiagonal matrix.
# On the other hand, similar(D, [neweltype,] shape...) should yield a sparse matrix.
# The first method below effects the former, and the second the latter.
similar(D::QuasiDiagonal, ::Type{T}) where {T} = QuasiDiagonal(similar(D.diag, T))
# The method below is moved to SparseArrays for now
# similar(D::QuasiDiagonal, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = spzeros(T, dims...)

copyto!(D1::QuasiDiagonal, D2::QuasiDiagonal) = (copyto!(D1.diag, D2.diag); D1)

size(D::QuasiDiagonal) = (cardinality(D.diag),cardinality(D.diag))
axes(D::QuasiDiagonal) = (axes(D.diag,1), axes(D.diag,1))

function size(D::QuasiDiagonal,d::Integer)
    if d<1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
    return d<=2 ? length(D.diag) : 1
end

@inline function getindex(D::QuasiDiagonal, i::Number, j::Number)
    @boundscheck checkbounds(D, i, j)
    if i == j
        @inbounds r = D.diag[i]
    else
        r = diagzero(D, i, j)
    end
    r
end
diagzero(::QuasiDiagonal{T},i,j) where {T} = zero(T)
diagzero(D::QuasiDiagonal{Matrix{T}},i,j) where {T} = zeros(T, size(D.diag[i], 1), size(D.diag[j], 2))

function setindex!(D::QuasiDiagonal, v, i::Int, j::Int)
    @boundscheck checkbounds(D, i, j)
    if i == j
        @inbounds D.diag[i] = v
    elseif !iszero(v)
        throw(ArgumentError("cannot set off-diagonal entry ($i, $j) to a nonzero value ($v)"))
    end
    return v
end


## structured matrix methods ##
function Base.replace_in_print_matrix(A::QuasiDiagonal,i::Integer,j::Integer,s::AbstractString)
    i==j ? s : Base.replace_with_centered_mark(s)
end

parent(D::QuasiDiagonal) = D.diag

ishermitian(D::QuasiDiagonal{<:Real}) = true
ishermitian(D::QuasiDiagonal{<:Number}) = isreal(D.diag)
ishermitian(D::QuasiDiagonal) = all(ishermitian, D.diag)
issymmetric(D::QuasiDiagonal{<:Number}) = true
issymmetric(D::QuasiDiagonal) = all(issymmetric, D.diag)
isposdef(D::QuasiDiagonal) = all(isposdef, D.diag)

factorize(D::QuasiDiagonal) = D

real(D::QuasiDiagonal) = QuasiDiagonal(real(D.diag))
imag(D::QuasiDiagonal) = QuasiDiagonal(imag(D.diag))

iszero(D::QuasiDiagonal) = all(iszero, D.diag)
isone(D::QuasiDiagonal) = all(isone, D.diag)
isdiag(D::QuasiDiagonal) = all(isdiag, D.diag)
isdiag(D::QuasiDiagonal{<:Number}) = true
istriu(D::QuasiDiagonal) = true
istril(D::QuasiDiagonal) = true
function triu!(D::QuasiDiagonal,k::Integer=0)
    n = size(D,1)
    if !(-n + 1 <= k <= n + 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n + 1) and at most $(n + 1) in an $n-by-$n matrix")))
    elseif k > 0
        fill!(D.diag,0)
    end
    return D
end

function tril!(D::QuasiDiagonal,k::Integer=0)
    n = size(D,1)
    if !(-n - 1 <= k <= n - 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n - 1) and at most $(n - 1) in an $n-by-$n matrix")))
    elseif k < 0
        fill!(D.diag,0)
    end
    return D
end

(==)(Da::QuasiDiagonal, Db::QuasiDiagonal) = Da.diag == Db.diag
(-)(A::QuasiDiagonal) = QuasiDiagonal(-A.diag)
(+)(Da::QuasiDiagonal, Db::QuasiDiagonal) = QuasiDiagonal(Da.diag + Db.diag)
(-)(Da::QuasiDiagonal, Db::QuasiDiagonal) = QuasiDiagonal(Da.diag - Db.diag)
(/)(D::QuasiDiagonal, x::Number) = QuasiDiagonal(D.diag / x)

conj(D::QuasiDiagonal) = QuasiDiagonal(conj(D.diag))
transpose(D::QuasiDiagonal{<:Number}) = D
transpose(D::QuasiDiagonal) = QuasiDiagonal(transpose.(D.diag))
adjoint(D::QuasiDiagonal{<:Number}) = conj(D)
adjoint(D::QuasiDiagonal) = QuasiDiagonal(adjoint.(D.diag))

function diag(D::QuasiDiagonal, k::Integer=0)
    # every branch call similar(..., ::Int) to make sure the
    # same vector type is returned independent of k
    if k == 0
        return copyto!(similar(D.diag, length(D.diag)), D.diag)
    elseif -size(D,1) <= k <= size(D,1)
        return fill!(similar(D.diag, size(D,1)-abs(k)), 0)
    else
        throw(ArgumentError(string("requested diagonal, $k, must be at least $(-size(D, 1)) ",
            "and at most $(size(D, 2)) for an $(size(D, 1))-by-$(size(D, 2)) matrix")))
    end
end
tr(D::QuasiDiagonal) = sum(D.diag)
det(D::QuasiDiagonal) = prod(D.diag)
logdet(D::QuasiDiagonal{<:Real}) = sum(log, D.diag)
function logdet(D::QuasiDiagonal{<:Complex}) # make sure branch cut is correct
    z = sum(log, D.diag)
    complex(real(z), rem2pi(imag(z), RoundNearest))
end

# Matrix functions
for f in (:exp, :log, :sqrt,
          :cos, :sin, :tan, :csc, :sec, :cot,
          :cosh, :sinh, :tanh, :csch, :sech, :coth,
          :acos, :asin, :atan, :acsc, :asec, :acot,
          :acosh, :asinh, :atanh, :acsch, :asech, :acoth)
    @eval $f(D::QuasiDiagonal) = QuasiDiagonal($f.(D.diag))
end
