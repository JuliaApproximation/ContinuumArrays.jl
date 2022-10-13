abstract type Basis{T} <: LazyQuasiMatrix{T} end
abstract type Weight{T} <: LazyQuasiVector{T} end


struct WeightLayout <: AbstractQuasiLazyLayout end
abstract type AbstractBasisLayout <: AbstractQuasiLazyLayout end
abstract type AbstractWeightedBasisLayout <: AbstractBasisLayout end
struct BasisLayout <: AbstractBasisLayout end
struct SubBasisLayout <: AbstractBasisLayout end
struct MappedBasisLayout <: AbstractBasisLayout end
struct WeightedBasisLayout{Basis} <: AbstractWeightedBasisLayout end
const SubWeightedBasisLayout = WeightedBasisLayout{SubBasisLayout}
const MappedWeightedBasisLayout = WeightedBasisLayout{MappedBasisLayout}

SubBasisLayouts = Union{SubBasisLayout,SubWeightedBasisLayout}
WeightedBasisLayouts = Union{WeightedBasisLayout,SubWeightedBasisLayout,MappedWeightedBasisLayout}
MappedBasisLayouts = Union{MappedBasisLayout,MappedWeightedBasisLayout}

struct AdjointBasisLayout{Basis} <: AbstractQuasiLazyLayout end
const AdjointSubBasisLayout = AdjointBasisLayout{SubBasisLayout}
const AdjointMappedBasisLayout = AdjointBasisLayout{MappedBasisLayout}

MemoryLayout(::Type{<:Basis}) = BasisLayout()
MemoryLayout(::Type{<:Weight}) = WeightLayout()

adjointlayout(::Type, ::Basis) where Basis<:AbstractBasisLayout = AdjointBasisLayout{Basis}()
broadcastlayout(::Type{typeof(*)}, ::WeightLayout, ::Basis) where Basis<:AbstractBasisLayout = WeightedBasisLayout{Basis}()

# A sub of a weight is still a weight
sublayout(::WeightLayout, _) = WeightLayout()
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{Map,AbstractVector}}) = MappedBasisLayout()


## Weighted basis interface
unweighted(P::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) = last(P.args)
unweighted(V::SubQuasiArray) = view(unweighted(parent(V)), parentindices(V)...)
weight(P::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) = first(P.args)
weight(V::SubQuasiArray) = weight(parent(V))[parentindices(V)[1]]

unweighted(a::AbstractQuasiArray) = unweighted(MemoryLayout(a), a)
# Default is lazy
ApplyStyle(::typeof(pinv), ::Type{<:Basis}) = LazyQuasiArrayApplyStyle()
pinv(J::Basis) = apply(pinv,J)


function _equals(::AbstractBasisLayout, ::AbstractBasisLayout, A, B)
    axes(A) == axes(B) && throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))
    false
end

_equals(::SubBasisLayouts, ::SubBasisLayouts, A::SubQuasiArray, B::SubQuasiArray) = parentindices(A) == parentindices(B) && parent(A) == parent(B)
_equals(::MappedBasisLayouts, ::MappedBasisLayouts, A::SubQuasiArray, B::SubQuasiArray) = parentindices(A) == parentindices(B) && demap(A) == demap(B)
_equals(::AbstractWeightedBasisLayout, ::AbstractWeightedBasisLayout, A, B) = weight(A) == weight(B) && unweighted(A) == unweighted(B)

@inline copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(+)}}) = +(broadcast(\,Ref(L.A),arguments(L.B))...)
@inline copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(+)},<:Any,<:AbstractQuasiVector}) =
    transform_ldiv(L.A, L.B)
for op in (:+, :-)
    @eval @inline copy(L::Ldiv{Lay,BroadcastLayout{typeof($op)},<:Any,<:AbstractQuasiVector}) where Lay<:MappedBasisLayouts =
        copy(Ldiv{Lay,LazyLayout}(L.A,L.B))
end

@inline function copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(-)}})
    a,b = arguments(L.B)
    (L.A\a)-(L.A\b)
end

@inline copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(-)},<:Any,<:AbstractQuasiVector}) =
    transform_ldiv(L.A, L.B)

@inline function copy(P::Ldiv{<:AbstractBasisLayout,<:AbstractBasisLayout})
    A, B = P.A, P.B
    A == B || throw(ArgumentError("Override copy for $(typeof(A)) \\ $(typeof(B))"))
    SquareEye{eltype(eltype(P))}((axes(A,2),)) # use double eltype for array-valued
end
@inline function copy(P::Ldiv{<:SubBasisLayouts,<:SubBasisLayouts})
    A, B = P.A, P.B
    parent(A) == parent(B) ||
        throw(ArgumentError("Override copy for $(typeof(A)) \\ $(typeof(B))"))
    Eye{eltype(eltype(P))}((axes(A,2),axes(B,2)))
end

@inline function copy(P::Ldiv{<:MappedBasisLayouts,<:MappedBasisLayouts})
    A, B = P.A, P.B
    demap(A)\demap(B)
end

function transform_ldiv_if_columns(P::Ldiv{<:MappedBasisLayouts,<:Any,<:Any,<:AbstractQuasiVector}, ::OneTo)
    A,B = P.A, P.B
    demap(A) \ B[invmap(basismap(A))]
end

function transform_ldiv_if_columns(P::Ldiv{<:MappedBasisLayouts,<:Any,<:Any,<:AbstractQuasiMatrix}, ::OneTo)
    A,B = P.A, P.B
    demap(A) \ B[invmap(basismap(A)),:]
end

copy(L::Ldiv{<:MappedBasisLayouts,ApplyLayout{typeof(*)}}) = copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{<:MappedBasisLayouts,ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = transform_ldiv(L.A, L.B)

@inline copy(L::Ldiv{<:AbstractBasisLayout,<:SubBasisLayouts}) = apply(\, L.A, ApplyQuasiArray(L.B))
@inline function copy(L::Ldiv{<:SubBasisLayouts,<:AbstractBasisLayout})
    P = parent(L.A)
    kr, jr = parentindices(L.A)
    layout_getindex(apply(\, P, L.B), jr, :) # avoid sparse arrays
end

# default to transform for expanding weights
copy(L::Ldiv{<:AbstractBasisLayout,WeightLayout}) = transform_ldiv(L.A, L.B)

# multiplication operators, reexpand in basis A
@inline function _broadcast_mul_ldiv(::Tuple{Any,AbstractBasisLayout}, A, B)
    a,b = arguments(B)
    @assert a isa AbstractQuasiVector # Only works for vec .* mat
    ab = (A * (A \ a)) .* b # broadcasted should be overloaded
    MemoryLayout(ab) isa BroadcastLayout && return transform_ldiv(A, ab)
    A \ ab
end

@inline function _broadcast_mul_ldiv(::Tuple{Any,ApplyLayout{typeof(*)}}, A, B)
    a,b = arguments(B)
    @assert a isa AbstractQuasiVector # Only works for vec .* mat
    args = arguments(*, b)
    *(A \ (a .* first(args)), tail(args)...)
end


function _broadcast_mul_ldiv(::Tuple{ScalarLayout,Any}, A, B)
    a,b = arguments(B)
    a * (A \ b)
end

function _broadcast_mul_ldiv(::Tuple{ScalarLayout,ApplyLayout{typeof(*)}}, A, B)
    a,b = arguments(B)
    a * (A \ b)
end

_broadcast_mul_ldiv(::Tuple{ScalarLayout,AbstractBasisLayout}, A, B) =
    _broadcast_mul_ldiv((ScalarLayout(),UnknownLayout()), A, B)
_broadcast_mul_ldiv(_, A, B) = copy(Ldiv{typeof(MemoryLayout(A)),UnknownLayout}(A,B))

copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(*)}}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)
copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)

# ambiguity
copy(L::Ldiv{<:MappedBasisLayouts,BroadcastLayout{typeof(*)}}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)
copy(L::Ldiv{<:MappedBasisLayouts,BroadcastLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)


# expansion
_grid(_, P) = error("Overload Grid")

_grid(::MappedBasisLayout, P) = invmap(parentindices(P)[1])[grid(demap(P))]
_grid(::SubBasisLayout, P) = grid(parent(P))
_grid(::WeightedBasisLayouts, P) = grid(unweighted(P))
grid(P) = _grid(MemoryLayout(P), P)


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
struct InvPlan{T, Fact, Dims} # <: Plan{T} We don't depend on AbstractFFTs
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


function plan_grid_transform(L, arr, dims=1:ndims(arr))
    p = grid(L)
    p, InvPlan(factorize(L[p,:]), dims)
end

plan_transform(P, arr, dims...) = plan_grid_transform(P, arr, dims...)[2]

_factorize(::AbstractBasisLayout, L, dims...; kws...) =
    TransformFactorization(plan_grid_transform(L, Array{eltype(L)}(undef, size(L,2), dims...), 1)...)



"""
    ProjectionFactorization(F, inds)

projects a factorization to a subset of coefficients. That is, if `P` is a `ProjectionFactorization`
then `P \\ f` is equivalent to `(F \\ f)[inds]`
"""
struct ProjectionFactorization{T, FAC<:Factorization{T}, INDS} <: Factorization{T}
    F::FAC
    inds::INDS
end

\(a::ProjectionFactorization, b::AbstractQuasiVector) = (a.F \ b)[a.inds]
\(a::ProjectionFactorization, b::AbstractQuasiMatrix) = (a.F \ b)[a.inds,:]



# if parent is finite dimensional default to its transform and project down
_sub_factorize(::Tuple{Any,Int}, (kr,jr), L, dims...; kws...) = ProjectionFactorization(factorize(parent(L), dims...; kws...), jr)
_sub_factorize(::Tuple{Any,Int}, (kr,jr)::Tuple{Any,OneTo}, L, dims...; kws...) = ProjectionFactorization(factorize(parent(L), dims...; kws...), jr)

# ∞-dimensional parents need to use transforms. For now we assume the size of the transform is equal to the size of the truncation
_sub_factorize(::Tuple{Any,Any}, (kr,jr)::Tuple{Any,OneTo}, L, dims...; kws...) =
    TransformFactorization(plan_grid_transform(parent(L), Array{eltype(L)}(undef, last(jr), dims...), 1)...)

# If jr is not OneTo we project
_sub_factorize(::Tuple{Any,Any}, (kr,jr), L, dims...; kws...) =
    ProjectionFactorization(factorize(parent(L)[:,OneTo(maximum(jr))]), jr)

_factorize(::SubBasisLayout, L, dims...; kws...) = _sub_factorize(size(parent(L)), parentindices(L), L, dims...; kws...)


"""
    MappedFactorization(F, map)

remaps a factorization to a different domain. That is, if `M` is a `MappedFactorization`
then `M \\ f` is equivalent to `F \\ f[map]`
"""
struct MappedFactorization{T, FAC<:Factorization{T}, MAP} <: Factorization{T}
    F::FAC
    map::MAP
end

\(a::MappedFactorization, b::AbstractQuasiVector) = a.F \ view(b, a.map)
\(a::MappedFactorization, b::AbstractVector) = a.F \ b
\(a::MappedFactorization, b::AbstractQuasiMatrix) = a.F \ view(b, a.map, :)


function invmap end

function _factorize(::MappedBasisLayout, L, dims...; kws...)
    kr, jr = parentindices(L)
    P = parent(L)
    MappedFactorization(factorize(view(P,:,jr), dims...; kws...), invmap(parentindices(L)[1]))
end

plan_ldiv(A, B::AbstractQuasiVector) = factorize(A)
plan_ldiv(A, B::AbstractQuasiMatrix) = factorize(A, size(B,2))

transform_ldiv(A::AbstractQuasiArray{T}, B::AbstractQuasiArray{V}, _) where {T,V} = plan_ldiv(A, B) \ B
transform_ldiv(A, B) = transform_ldiv(A, B, size(A))


"""
    transform(A, f)

finds the coefficients of a function `f` expanded in a basis defined as the columns of a quasi matrix `A`.
It is equivalent to
```
A \\ f.(axes(A,1))
```
"""
transform(A, f) = A \ f.(axes(A,1))

"""
    expand(A, f)

expands a function `f` im a basis defined as the columns of a quasi matrix `A`.
It is equivalent to
```
A / A \\ f.(axes(A,1))
```
"""
expand(A, f) = A * transform(A, f)

copy(L::Ldiv{<:AbstractBasisLayout}) = transform_ldiv(L.A, L.B)
# TODO: redesign to use simplifiable(\, A, B)
copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = transform_ldiv(L.A, L.B)
copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(*)}}) = copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(L.A, L.B))
# A BroadcastLayout of unknown function is only knowable pointwise
transform_ldiv_if_columns(L, _) = ApplyQuasiArray(\, L.A, L.B)
transform_ldiv_if_columns(L, ::OneTo) = transform_ldiv(L.A,L.B)
transform_ldiv_if_columns(L) = transform_ldiv_if_columns(L, axes(L.B,2))
copy(L::Ldiv{<:AbstractBasisLayout,<:BroadcastLayout}) = transform_ldiv_if_columns(L)
# Inclusion are QuasiArrayLayout
copy(L::Ldiv{<:AbstractBasisLayout,QuasiArrayLayout}) = transform_ldiv(L.A, L.B)
# Otherwise keep lazy to support, e.g., U\D*T
copy(L::Ldiv{<:AbstractBasisLayout,<:AbstractLazyLayout}) = transform_ldiv_if_columns(L)
copy(L::Ldiv{<:AbstractBasisLayout,ZerosLayout}) = Zeros{eltype(L)}(axes(L)...)

transform_ldiv_if_columns(L::Ldiv{<:Any,<:ApplyLayout{typeof(hcat)}}, ::OneTo) = transform_ldiv(L.A, L.B)
transform_ldiv_if_columns(L::Ldiv{<:Any,<:ApplyLayout{typeof(hcat)}}, _) = hcat((Ref(L.A) .\ arguments(hcat, L.B))...)

"""
    WeightedFactorization(w, F)

weights a factorization `F` by `w`.
"""
struct WeightedFactorization{T, WW, FAC<:Factorization{T}} <: Factorization{T}
    w::WW
    F::FAC
end

_factorize(::WeightedBasisLayouts, wS, dims...; kws...) = WeightedFactorization(weight(wS), factorize(unweighted(wS), dims...; kws...))


\(F::WeightedFactorization, b::AbstractQuasiVector) = F.F \ (b ./ F.w)

##
# Algebra
##

struct ExpansionLayout{Lay} <: AbstractLazyLayout end
applylayout(::Type{typeof(*)}, ::Lay, ::Union{PaddedLayout,AbstractStridedLayout}) where Lay <: AbstractBasisLayout = ExpansionLayout{Lay}()

basis(v::ApplyQuasiArray{<:Any,N,typeof(*)}) where N = v.args[1]
coefficients(v::ApplyQuasiArray{<:Any,N,typeof(*),<:Tuple{Any,Any}}) where N = v.args[2]


function unweighted(lay::ExpansionLayout, a)
    wP,c = arguments(lay, a)
    unweighted(wP) * c
end

LazyArrays._mul_arguments(::ExpansionLayout, A) = LazyArrays._mul_arguments(ApplyLayout{typeof(*)}(), A)
copy(L::Ldiv{Bas,<:ExpansionLayout}) where Bas<:AbstractBasisLayout = copy(Ldiv{Bas,ApplyLayout{typeof(*)}}(L.A, L.B))
copy(L::Mul{<:ExpansionLayout,Lay}) where Lay = copy(Mul{ApplyLayout{typeof(*)},Lay}(L.A, L.B))

function _broadcastbasis(::typeof(+), _, _, a, b)
    try
        a ≠ b && error("Overload broadcastbasis(::typeof(+), ::$(typeof(a)), ::$(typeof(b)))")
    catch
        error("Overload broadcastbasis(::typeof(+), ::$(typeof(a)), ::$(typeof(b)))")
    end
    a
end

_broadcastbasis(::typeof(+), ::MappedBasisLayouts, ::MappedBasisLayouts, a, b) = broadcastbasis(+, demap(a), demap(b))[basismap(a), :]
function _broadcastbasis(::typeof(+), ::SubBasisLayout, ::SubBasisLayout, a, b)
    kr_a,jr_a = parentindices(a)
    kr_b,jr_b = parentindices(b)
    @assert kr_a == kr_b # frist axes must match
    view(broadcastbasis(+, parent(a), parent(b)), kr_a, union(jr_a,jr_b))
end
_broadcastbasis(::typeof(+), ::SubBasisLayout, _, a, b) = broadcastbasis(+, parent(a), b)
_broadcastbasis(::typeof(+), _, ::SubBasisLayout, a, b) = broadcastbasis(+, a, parent(b))

broadcastbasis(::typeof(+), a, b) = _broadcastbasis(+, MemoryLayout(a), MemoryLayout(b), a, b)
broadcastbasis(::typeof(+), a, b, c...) = broadcastbasis(+, broadcastbasis(+, a, b), c...)

broadcastbasis(::typeof(-), a, b) = broadcastbasis(+, a, b)

@eval function layout_broadcasted(::NTuple{2,ExpansionLayout}, ::typeof(-), f, g)
    S,c = arguments(f)
    T,d = arguments(g)
    ST = broadcastbasis(-, S, T)
    ST * ((ST \ S) * c - (ST \ T) * d)
end

_plus_P_ldiv_Ps_cs(P, ::Tuple{}, ::Tuple{}) = ()
_plus_P_ldiv_Ps_cs(P, Q::Tuple, cs::Tuple) = tuple((P \ first(Q)) * first(cs), _plus_P_ldiv_Ps_cs(P, tail(Q), tail(cs))...)
function layout_broadcasted(::Tuple{Vararg{ExpansionLayout}}, ::typeof(+), fs...)
    Ps = first.(arguments.(fs))
    cs = last.(arguments.(fs))
    P = broadcastbasis(+, Ps...)
    P * +(_plus_P_ldiv_Ps_cs(P, Ps, cs)...)  # +((Ref(P) .\ Ps .* cs)...)
end

function layout_broadcasted(::NTuple{2,ExpansionLayout}, ::typeof(*), a, f)
    axes(a,1) == axes(f,1) || throw(DimensionMismatch())
    P,c = arguments(f)
    (a .* P) * c
end

function layout_broadcasted(::Tuple{ExpansionLayout{<:AbstractWeightedBasisLayout},AbstractBasisLayout}, ::typeof(*), a, P)
    axes(a,1) == axes(P,1) || throw(DimensionMismatch())
    wQ,c = arguments(a)
    w,Q = weight(wQ),unweighted(wQ)
    uP = unweighted(a) .* P
    w .* uP
end

# function layout_broadcasted(::Tuple{PolynomialLayout,MappedBasisLayouts}, ::typeof(*), x, P)
#     axes(a,1) == axes(P,1) || throw(DimensionMismatch())
#     wQ,c = arguments(a)
#     w,Q = arguments(wQ)
#     uP = unweighted(a) .* P
#     w .* uP
# end


function _equals(::ExpansionLayout, ::ExpansionLayout, f, g)
    S,c = arguments(f)
    T,d = arguments(g)
    ST = broadcastbasis(+, S, T)
    (ST \ S) * c == (ST \ T) * d
end

function QuasiArrays._mul_summary(::ExpansionLayout, io::IO, f)
    P,c = arguments(f)
    show(io, P)
    print(io, " * ")
    show(io, c)
end

##
# Multiplication for mapped and subviews x .* view(P,...)
##

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    T = promote_type(eltype(x), eltype(C))
    x == axes(C,1) || throw(DimensionMismatch())
    P = parent(C)
    kr,jr = parentindices(C)
    y = axes(P,1)
    Y = P \ (y .* P)
    X = kr.A \ (Y     - kr.b * I)
    P[kr, :] * view(X,:,jr)
end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}})
    T = promote_type(eltype(x), eltype(C))
    x == axes(C,1) || throw(DimensionMismatch())
    P = parent(C)
    kr,_ = parentindices(C)
    y = axes(P,1)
    Y = P \ (y .* P)
    X = kr.A \ (Y     - kr.b * I)
    P[kr, :] * X
end


# function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), f::AbstractQuasiVector, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}})
#     T = promote_type(eltype(f), eltype(C))
#     axes(f,1) == axes(C,1) || throw(DimensionMismatch())
#     P = parent(C)
#     kr,jr = parentindices(C)
#     (f[invmap(kr)] .* P)[kr,jr]
# end

# function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), f::AbstractQuasiVector, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
#     T = promote_type(eltype(f), eltype(C))
#     axes(f,1) == axes(C,1) || throw(DimensionMismatch())
#     P = parent(C)
#     kr,jr = parentindices(C)
#     (f[invmap(kr)] .* P)[kr,jr]
# end

broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), f::Broadcasted, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}) =
    broadcast(*, materialize(f), C)

function layout_broadcasted(::Tuple{PolynomialLayout,WeightedBasisLayout}, ::typeof(*), x, C)
    x == axes(C,1) || throw(DimensionMismatch())
    weight(C) .* (x .* unweighted(C))
end


## materialize views

# materialize(S::SubQuasiArray{<:Any,2,<:ApplyQuasiArray{<:Any,2,typeof(*),<:Tuple{<:Basis,<:Any}}}) =
#     *(arguments(S)...)



# mass matrix
# y = p(x), dy = p'(x) * dx
# \int_a^b f(y) g(y) dy = \int_{-1}^1 f(p(x))*g(p(x)) * p'(x) dx


_sub_getindex(A, kr, jr) = A[kr, jr]
_sub_getindex(A, ::Slice, ::Slice) = A

@simplify function *(Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}},
             B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    A = Ac'
    PA,PB = parent(A),parent(B)
    kr,jr = parentindices(B)
    _sub_getindex((PA'PB)/kr.A,parentindices(A)[2],jr)
end


# Differentiation of sub-arrays

# avoid stack overflow from unmaterialize Derivative() * parent()
_der_sub(DP, inds...) = DP[inds...]
_der_sub(DP::ApplyQuasiMatrix{T,typeof(*),<:Tuple{Derivative,Any}}, kr, jr) where T = ApplyQuasiMatrix{T}(*, DP.args[1], view(DP.args[2], kr, jr))

# need to customise simplifiable so can't use @simplify
simplifiable(::typeof(*), A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}})= simplifiable(*, Derivative(axes(parent(B),1)), parent(B))
simplifiable(::typeof(*), Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = simplifiable(*, Bc', Ac')
function mul(A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}})
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    P = parent(B)
    _der_sub(Derivative(axes(P,1))*P, parentindices(B)...)
end
mul(Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = mul(Bc', Ac')'

simplifiable(::typeof(*), A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}) = simplifiable(*, Derivative(axes(parent(B),1)), parent(B))
simplifiable(::typeof(*), Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = simplifiable(*, Bc', Ac')
function mul(A::Derivative, B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    P = parent(B)
    kr,jr = parentindices(B)
    (Derivative(axes(P,1))*P*kr.A)[kr,jr]
end
mul(Ac::QuasiAdjoint{<:Any,<:SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}}, Bc::QuasiAdjoint{<:Any,<:Derivative}) = mul(Bc', Ac')'

# we represent as a Mul with a banded matrix
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractVector}}) = SubBasisLayout()
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:AbstractVector}}) = MappedBasisLayout()
sublayout(::WeightedBasisLayouts, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:AbstractVector}}) = MappedWeightedBasisLayout()
sublayout(::WeightedBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractVector}}) = SubWeightedBasisLayout()
sublayout(::MappedWeightedBasisLayout, ::Type{<:Tuple{<:Inclusion,<:AbstractVector}}) = MappedWeightedBasisLayout()

@inline sub_materialize(::AbstractBasisLayout, V::AbstractQuasiArray) = V
@inline sub_materialize(::AbstractBasisLayout, V::AbstractArray) = V

demap(x) = x
demap(x::BroadcastQuasiArray) = BroadcastQuasiArray(x.f, map(demap, arguments(x))...)
demap(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{Any,Slice}}) = parent(V)
demap(V::SubQuasiArray{<:Any,1}) = parent(V)
function demap(V::SubQuasiArray{<:Any,2})
    kr, jr = parentindices(V)
    demap(parent(V)[kr,:])[:,jr]
end
function demap(wB::ApplyQuasiArray{<:Any,N,typeof(*)}) where N
    a = arguments(wB)
    *(demap(first(a)), tail(a)...)
end


basismap(x::SubQuasiArray) = parentindices(x)[1]
basismap(x::BroadcastQuasiArray) = basismap(x.args[1])


##
# SubLayout behaves like ApplyLayout{typeof(*)}

combine_mul_styles(::SubBasisLayouts) = combine_mul_styles(ApplyLayout{typeof(*)}())
_mul_arguments(::SubBasisLayouts, A) = _mul_arguments(ApplyLayout{typeof(*)}(), A)
arguments(::SubBasisLayouts, A) = arguments(ApplyLayout{typeof(*)}(), A)
call(::SubBasisLayouts, ::SubQuasiArray) = *

combine_mul_styles(::AdjointSubBasisLayout) = combine_mul_styles(ApplyLayout{typeof(*)}())
_mul_arguments(::AdjointSubBasisLayout, A) = _mul_arguments(ApplyLayout{typeof(*)}(), A)
arguments(::AdjointSubBasisLayout, A) = arguments(ApplyLayout{typeof(*)}(), A)
call(::AdjointSubBasisLayout, ::SubQuasiArray) = *

copy(M::Mul{AdjointSubBasisLayout,<:SubBasisLayouts}) = apply(*, arguments(M.A)..., arguments(M.B)...)

function arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractUnitRange}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = _BandedMatrix(Ones{Int}((Base.OneTo(1),axes(jr,1))), axes(A,2), first(jr)-1,1-first(jr))
    A,P
end

function arguments(::ApplyLayout{typeof(*)}, V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:Inclusion,<:AbstractVector}})
    A = parent(V)
    _,jr = parentindices(V)
    first(jr) ≥ 1 || throw(BoundsError())
    P = Eye{Int}((axes(A,2),))[:,jr]
    A,P
end

####
# sum
####


function __sum(::SubBasisLayout, Vm, dims)
    @assert dims == 1
    sum(parent(Vm); dims=dims)[:,parentindices(Vm)[2]]
end

__sum(::AdjointBasisLayout, Vm::AbstractQuasiMatrix, dims) = permutedims(sum(Vm'; dims=(isone(dims) ? 2 : 1)))


function __sum(::MappedBasisLayouts, V, dims)
    kr = basismap(V)
    @assert kr isa AbstractAffineQuasiVector
    sum(demap(V); dims=dims)/kr.A
end

__sum(::ExpansionLayout, A, dims) = __sum(ApplyLayout{typeof(*)}(), A, dims)
__cumsum(::ExpansionLayout, A, dims) = __cumsum(ApplyLayout{typeof(*)}(), A, dims)

include("basisconcat.jl")
include("basiskron.jl")
include("splines.jl")
