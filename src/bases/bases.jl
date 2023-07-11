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

# copy with an Inclusion can not be materialized
copy(V::SubQuasiArray{<:Any,N,<:Basis,<:Tuple{AbstractQuasiVector,Vararg{Any}}, trfl}) where {N,trfl}  = V


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
_grid(_, P, n...) = error("Overload Grid")

_grid(::MappedBasisLayout, P, n...) = invmap(parentindices(P)[1])[grid(demap(P), n...)]
_grid(::SubBasisLayout, P::AbstractQuasiMatrix, n) = grid(parent(P), maximum(parentindices(P)[2][n]))
_grid(::SubBasisLayout, P::AbstractQuasiMatrix) = grid(parent(P), maximum(parentindices(P)[2]))
_grid(::WeightedBasisLayouts, P, n...) = grid(unweighted(P), n...)


"""
    grid(P, n...)

Creates a grid of points. if `n` is unspecified it will
be sufficient number of points to determine `size(P,2)`
coefficients. Otherwise its enough points to determine `n`
coefficients.
"""
grid(P, n...) = _grid(MemoryLayout(P), P, n...)


# values(f) = 



function plan_grid_transform(lay, L, szs::NTuple{N,Int}, dims=1:N) where N
    p = grid(L)
    p, InvPlan(factorize(L[p,:]), dims)
end

function plan_grid_transform(::MappedBasisLayout, L, szs::NTuple{N,Int}, dims=1:N) where N
    x,F = plan_grid_transform(demap(L), szs, dims)
    invmap(parentindices(L)[1])[x], F
end

plan_grid_transform(L, szs::NTuple{N,Int}, dims=1:N) where N = plan_grid_transform(MemoryLayout(L), L, szs, dims)

plan_grid_transform(L, arr::AbstractArray{<:Any,N}, dims=1:N) where N = 
    plan_grid_transform(L, size(arr), dims)

plan_transform(P, szs, dims...) = plan_grid_transform(P, szs, dims...)[2]

_factorize(::AbstractBasisLayout, L, dims...; kws...) =
    TransformFactorization(plan_grid_transform(L, (size(L,2), dims...), 1)...)



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
    TransformFactorization(plan_grid_transform(parent(L), (last(jr), dims...), 1)...)

# If jr is not OneTo we project
_sub_factorize(::Tuple{Any,Any}, (kr,jr), L, dims...; kws...) =
    ProjectionFactorization(factorize(parent(L)[:,OneTo(maximum(jr))], dims...), jr)

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

"""
    expand(v)

finds a natural basis for a quasi-vector and expands
in that basis.
"""
function expand(v)
    P = basis(v)
    ApplyQuasiArray(*, P, P \ v)
end



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
applylayout(::Type{typeof(*)}, ::Lay, ::Union{PaddedLayout,AbstractStridedLayout,ZerosLayout}) where Lay <: AbstractBasisLayout = ExpansionLayout{Lay}()

"""
    basis(v)

gives a basis for expanding given quasi-vector.
"""
basis(v) = basis_layout(MemoryLayout(v), v)

basis_layout(::ExpansionLayout, v::ApplyQuasiArray{<:Any,N,typeof(*)}) where N = v.args[1]
basis_layout(lay, v) = basis_axes(axes(v,1), v) # allow choosing a basis based on axes
basis_axes(ax, v) = error("Overload for $ax")

coefficients(v::ApplyQuasiArray{<:Any,N,typeof(*),<:Tuple{Any,Any}}) where N = v.args[2]
coefficients(v::ApplyQuasiArray{<:Any,N,typeof(*),<:Tuple{Any,Any,Vararg{Any}}}) where N = ApplyArray(*, tail(v.args)...)


function unweighted(lay::ExpansionLayout, a)
    wP,c = arguments(lay, a)
    unweighted(wP) * c
end

LazyArrays._mul_arguments(::ExpansionLayout, A) = LazyArrays._mul_arguments(ApplyLayout{typeof(*)}(), A)
copy(L::Ldiv{Bas,<:ExpansionLayout}) where Bas<:AbstractBasisLayout = copy(Ldiv{Bas,ApplyLayout{typeof(*)}}(L.A, L.B))
copy(L::Mul{<:ExpansionLayout,Lay}) where Lay = copy(Mul{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
copy(L::Mul{<:ExpansionLayout,Lay}) where Lay<:AbstractLazyLayout = copy(Mul{ApplyLayout{typeof(*)},Lay}(L.A, L.B))

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

function layout_broadcasted(::Tuple{Any,ExpansionLayout}, ::typeof(*), a, f)
    axes(a)[1] == axes(f)[1] || throw(DimensionMismatch())
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

# need to customise simplifiable so can't use @simplify
function diff(B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:Inclusion,<:Any}}; dims::Integer)
    @assert dims == 1
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    P = parent(B)
    diff(P, dims)[parentindices(B)...]
end

function diff(B::SubQuasiArray{<:Any,2,<:AbstractQuasiMatrix,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}; dims::Integer)
    @assert dims == 1
    axes(A,2) == axes(B,1) || throw(DimensionMismatch())
    P = parent(B)
    kr,jr = parentindices(B)
    (diff(P, dims)*kr.A)[kr,jr]
end

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
function sum_layout(::SubBasisLayout, Vm, dims)
    @assert dims == 1
    sum(parent(Vm); dims=dims)[:,parentindices(Vm)[2]]
end

sum_layout(::AdjointBasisLayout, Vm::AbstractQuasiMatrix, dims) = permutedims(sum(Vm'; dims=(isone(dims) ? 2 : 1)))


function sum_layout(::MappedBasisLayouts, V, dims)
    kr = basismap(V)
    @assert kr isa AbstractAffineQuasiVector
    sum(demap(V); dims=dims)/kr.A
end

sum_layout(::ExpansionLayout, A, dims) = sum_layout(ApplyLayout{typeof(*)}(), A, dims)
cumsum_layout(::ExpansionLayout, A, dims) = cumsum_layout(ApplyLayout{typeof(*)}(), A, dims)

include("basisconcat.jl")
include("basiskron.jl")
include("splines.jl")
