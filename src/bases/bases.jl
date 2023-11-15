abstract type Basis{T} <: LazyQuasiMatrix{T} end
abstract type Weight{T} <: LazyQuasiVector{T} end


abstract type AbstractWeightLayout <: AbstractQuasiLazyLayout end
struct WeightLayout <: AbstractWeightLayout end
struct MappedWeightLayout <: AbstractWeightLayout end
abstract type AbstractBasisLayout <: AbstractQuasiLazyLayout end
abstract type AbstractWeightedBasisLayout <: AbstractBasisLayout end
struct BasisLayout <: AbstractBasisLayout end
struct SubBasisLayout <: AbstractBasisLayout end
struct MappedBasisLayout <: AbstractBasisLayout end
struct WeightedBasisLayout{Basis} <: AbstractWeightedBasisLayout end
const SubWeightedBasisLayout = WeightedBasisLayout{SubBasisLayout}
const MappedWeightedBasisLayout = WeightedBasisLayout{MappedBasisLayout}

struct AdjointBasisLayout{Basis} <: AbstractQuasiLazyLayout end
const AdjointSubBasisLayout = AdjointBasisLayout{SubBasisLayout}

SubBasisLayouts = Union{SubBasisLayout,SubWeightedBasisLayout}
WeightedBasisLayouts = Union{WeightedBasisLayout,SubWeightedBasisLayout,MappedWeightedBasisLayout}
MappedBasisLayouts = Union{MappedBasisLayout,MappedWeightedBasisLayout}
AdjointMappedBasisLayouts = AdjointBasisLayout{<:MappedBasisLayouts}

MemoryLayout(::Type{<:Basis}) = BasisLayout()
MemoryLayout(::Type{<:Weight}) = WeightLayout()

adjointlayout(::Type, ::Basis) where Basis<:AbstractBasisLayout = AdjointBasisLayout{Basis}()
broadcastlayout(::Type{typeof(*)}, ::AbstractWeightLayout, ::Basis) where Basis<:AbstractBasisLayout = WeightedBasisLayout{Basis}()

sublayout(::AbstractWeightLayout, _) = WeightLayout()
sublayout(::AbstractWeightLayout, ::Type{<:Tuple{Map}}) = MappedWeightLayout()
sublayout(::AbstractBasisLayout, ::Type{<:Tuple{Map,AbstractVector}}) = MappedBasisLayout()

# copy with an Inclusion can not be materialized
copy(V::SubQuasiArray{<:Any,N,<:Basis,<:Tuple{AbstractQuasiVector,Vararg{Any}}, trfl}) where {N,trfl}  = V


## Weighted basis interface
unweighted(P::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) = last(P.args)
unweighted(V::SubQuasiArray) = view(unweighted(parent(V)), parentindices(V)...)
weight(P::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{AbstractQuasiVector,AbstractQuasiMatrix}}) = first(P.args)
weight(V::SubQuasiArray) = weight(parent(V))[parentindices(V)[1]]
weight(V::SubQuasiArray{<:Any,2,<:Any, <:Tuple{Inclusion,Any}}) = weight(parent(V))

unweighted(a::AbstractQuasiArray) = unweighted(MemoryLayout(a), a)
# Default is lazy
ApplyStyle(::typeof(pinv), ::Type{<:Basis}) = LazyQuasiArrayApplyStyle()
pinv(J::Basis) = apply(pinv,J)


function equals_layout(::AbstractBasisLayout, ::AbstractBasisLayout, A, B)
    axes(A) == axes(B) && throw(ArgumentError("Override == to compare bases of type $(typeof(A)) and $(typeof(B))"))
    false
end

equals_layout(::SubBasisLayouts, ::SubBasisLayouts, A::SubQuasiArray, B::SubQuasiArray) = parentindices(A) == parentindices(B) && parent(A) == parent(B)
equals_layout(::MappedBasisLayouts, ::MappedBasisLayouts, A::SubQuasiArray, B::SubQuasiArray) = parentindices(A) == parentindices(B) && demap(A) == demap(B)
equals_layout(::AbstractWeightedBasisLayout, ::AbstractWeightedBasisLayout, A, B) = weight(A) == weight(B) && unweighted(A) == unweighted(B)

for op in (:+, :-)
    @eval begin
        @inline copy(L::Ldiv{<:AbstractBasisLayout,BroadcastLayout{typeof($op)}}) = basis_broadcast_ldiv_size($op, size(L), L.A, L.B)
        @inline copy(L::Ldiv{<:MappedBasisLayouts,BroadcastLayout{typeof($op)}}) = copy(Ldiv{BasisLayout,BroadcastLayout{typeof($op)}}(L.A, L.B))
        basis_broadcast_ldiv_size(::typeof($op), ::Tuple{Integer}, A, B) = transform_ldiv(A, B)
    end
end

basis_broadcast_ldiv_size(::typeof(+), _, A, B) = +(broadcast(\,Ref(A),arguments(B))...)



@inline function basis_broadcast_ldiv_size(::typeof(-), _, A, B)
    a,b = arguments(B)
    (A\a)-(A\b)
end


# TODO: remove as Not type stable
simplifiable(L::Ldiv{<:AbstractBasisLayout,<:AbstractBasisLayout}) = Val(L.A == L.B)
@inline function copy(P::Ldiv{<:AbstractBasisLayout,<:AbstractBasisLayout})
    A, B = P.A, P.B
    A == B || throw(ArgumentError("Override copy for $(typeof(A)) \\ $(typeof(B))"))
    SquareEye{eltype(eltype(P))}((axes(A,2),)) # use double eltype for array-valued
end

simplifiable(L::Ldiv{<:SubBasisLayouts,<:SubBasisLayouts}) = Val(parent(L.A) == parent(L.B))
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

copy(P::Ldiv{<:MappedBasisLayouts}) = mapped_ldiv_size(size(P), P.A, P.B)
copy(P::Ldiv{<:MappedBasisLayouts, <:AbstractLazyLayout}) = mapped_ldiv_size(size(P), P.A, P.B)
copy(P::Ldiv{<:MappedBasisLayouts, <:AbstractBasisLayout}) = mapped_ldiv_size(size(P), P.A, P.B)
@inline copy(L::Ldiv{<:MappedBasisLayouts,ApplyLayout{typeof(hcat)}}) = mapped_ldiv_size(size(L), L.A, L.B)
copy(P::Ldiv{<:MappedBasisLayouts, ApplyLayout{typeof(*)}}) = copy(Ldiv{BasisLayout,ApplyLayout{typeof(*)}}(P.A, P.B))

mapped_ldiv_size(::Tuple{Integer}, A, B) = demap(A) \ B[invmap(basismap(A))]
mapped_ldiv_size(::Tuple{Integer,Int}, A, B) = demap(A) \ B[invmap(basismap(A)),:]
mapped_ldiv_size(::Tuple{Integer,Any}, A, B) = copy(Ldiv{BasisLayout,typeof(MemoryLayout(B))}(A, B))

# following allows us to use simplification
@inline copy(L::Ldiv{Lay,<:SubBasisLayouts}) where Lay<:AbstractBasisLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline function copy(L::Ldiv{<:SubBasisLayouts,<:AbstractBasisLayout})
    P = parent(L.A)
    kr, jr = parentindices(L.A)
    layout_getindex(apply(\, P, L.B), jr, :) # avoid sparse arrays
end

# default to transform for expanding weights
copy(L::Ldiv{<:AbstractBasisLayout,<:AbstractWeightLayout}) = transform_ldiv(L.A, L.B)

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
copy(L::Ldiv{<:MappedBasisLayouts,BroadcastLayout{typeof(*)}}) = _broadcast_mul_ldiv(map(MemoryLayout,arguments(L.B)), L.A, L.B)





"""
    grid(P, n...)

Creates a grid of points. if `n` is unspecified it will
be sufficient number of points to determine `size(P,2)`
coefficients. If `n` is an integer or `Block` its enough points to determine `n`
coefficients. If `n` is a tuple then it returns a tuple of grids corresponding to a
tensor-product. That is, a 5⨱6 2D transform would be
```julia
(x,y) = grid(P, (5,6))
plan_transform(P, (5,6)) * f.(x, y')
```
and a 5×6×7 3D transform would be
```julia
(x,y,z) = grid(P, (5,6,7))
plan_transform(P, (5,6,7)) * f.(x, y', reshape(z,1,1,:))
```
"""
grid(P, n::Block{1}) = grid_layout(MemoryLayout(P), P, n...)
grid(P, n::Integer) = grid_layout(MemoryLayout(P), P, n...)
grid(L, B::Block) = grid(L, Block.(B.n)) # grid(L, Block(2,3)) == grid(L, (Block(2), Block(3))
grid(L, ns::Tuple) = grid.(Ref(L), ns)
grid(L) = grid(L, size(L,2))

grid_layout(_, P, n) = grid_axis(axes(P,2), P, n)

grid_axis(::OneTo, P, n::Block) = grid(P, size(P,2))

grid_layout(::MappedBasisLayout, P, n) = invmap(parentindices(P)[1])[grid(demap(P), n)]
grid_layout(::SubBasisLayout, P::AbstractQuasiMatrix, n) = grid(parent(P), parentindices(P)[2][n])
grid_layout(::WeightedBasisLayouts, P, n) = grid(unweighted(P), n)


# Default transform is just solve least squares on a grid
# note this computes the grid twice.
function plan_transform_layout(lay, L, szs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N
    ps = grid(L, getindex.(Ref(szs), dims))
    if dims isa Integer
        InvPlan(factorize(L[ps,:]), dims)
    else
        InvPlan(map(p -> factorize(L[p,:]), ps), dims)
    end
end


plan_transform_layout(::MappedBasisLayout, L, szs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N = plan_transform(demap(L), szs, dims)
plan_transform(L, szs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N = plan_transform_layout(MemoryLayout(L), L, szs, dims)

plan_transform(L, arr::AbstractArray, dims...) = plan_transform(L, size(arr), dims...)
plan_transform(L, lng::Union{Integer,Block{1}}, dims...) = plan_transform(L, (lng,), dims...)
plan_transform(L) = plan_transform(L, size(L,2))
    


plan_grid_transform(P, szs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N = grid(P, getindex.(Ref(szs), dims)), plan_transform(P, szs, dims)
function plan_grid_transform(P, lng::Union{Integer,Block{1}}, dims=1)
    @assert dims == 1
    grid(P, lng), plan_transform(P, lng, dims)
end

_factorize(::AbstractBasisLayout, L, dims...; kws...) = TransformFactorization(plan_grid_transform(L, (size(L,2), dims...), 1)...)


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

transform_ldiv_size(_, A::AbstractQuasiArray{T}, B::AbstractQuasiArray{V}) where {T,V} = plan_ldiv(A, B) \ B
transform_ldiv(A, B) = transform_ldiv_size(size(A), A, B)


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
    ApplyQuasiArray(*, P, tocoefficients(P \ v))
end





@inline copy(L::Ldiv{<:AbstractBasisLayout}) = basis_ldiv_size(size(L), L.A, L.B)
@inline copy(L::Ldiv{<:AbstractBasisLayout,<:AbstractLazyLayout}) = basis_ldiv_size(size(L), L.A, L.B)
@inline function copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(*)}})
    simplifiable(\, L.A, first(arguments(*, L.B))) isa Val{true} && return copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(L.A, L.B))
    basis_ldiv_size(size(L), L.A, L.B)
end
@inline copy(L::Ldiv{<:AbstractBasisLayout,ZerosLayout}) = Zeros{eltype(L)}(axes(L)...)

@inline basis_ldiv_size(_, A, B) = copy(Ldiv{UnknownLayout,typeof(MemoryLayout(B))}(A, B))
@inline basis_ldiv_size(::Tuple{Integer}, A, B) = transform_ldiv(A, B)
@inline basis_ldiv_size(::Tuple{Integer,Int}, A, B) = transform_ldiv(A, B)

@inline copy(L::Ldiv{<:AbstractBasisLayout,ApplyLayout{typeof(hcat)}}) = basis_hcat_ldiv_size(size(L), L.A, L.B)
@inline basis_hcat_ldiv_size(::Tuple{Integer,Int}, A, B) = transform_ldiv(A, B)
@inline basis_hcat_ldiv_size(_, A, B) = hcat((Ref(A) .\ arguments(hcat, B))...)


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
const CoefficientLayouts = Union{PaddedLayout,AbstractStridedLayout,ZerosLayout}
applylayout(::Type{typeof(*)}, ::Lay, ::CoefficientLayouts) where Lay <: AbstractBasisLayout = ExpansionLayout{Lay}()

tocoefficients(v) = tocoefficients_layout(MemoryLayout(v), v)
tocoefficients_layout(::CoefficientLayouts, v) = v
tocoefficients_layout(_, v) = tocoefficients_size(size(v), v)
tocoefficients_size(::NTuple{N,Int}, v) where N = Array(v)
tocoefficients_size(_, v) = v # the default is to leave it, even though we aren't technically making an ExpansionLayout

"""
    basis(v)

gives a basis for expanding given quasi-vector.
"""
basis(v) = basis_layout(MemoryLayout(v), v)

basis_layout(::ExpansionLayout, v::ApplyQuasiArray{<:Any,N,typeof(*)}) where N = v.args[1]
basis_layout(lay::ApplyLayout{typeof(*)}, v) = basis(first(arguments(lay, v)))
basis_layout(lay::AbstractBasisLayout, v) = v
basis_layout(lay, v) = basis_axes(axes(v,1), v) # allow choosing a basis based on axes
basis_axes(ax, v) = error("Overload for $ax")

coefficients(v::ApplyQuasiArray{<:Any,N,typeof(*),<:Tuple{Any,Any}}) where N = v.args[2]
coefficients(v::ApplyQuasiArray{<:Any,N,typeof(*),<:Tuple{Any,Any,Vararg{Any}}}) where N = ApplyArray(*, tail(v.args)...)


function unweighted(lay::ExpansionLayout, a)
    wP,c = arguments(lay, a)
    unweighted(wP) * c
end

LazyArrays._mul_arguments(::ExpansionLayout, A) = LazyArrays._mul_arguments(ApplyLayout{typeof(*)}(), A)
copy(L::Ldiv{Lay,<:ExpansionLayout}) where Lay<:AbstractBasisLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
copy(L::Ldiv{Lay,<:ExpansionLayout}) where Lay<:MappedBasisLayouts = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
copy(L::Mul{<:ExpansionLayout,Lay}) where Lay = copy(Mul{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
copy(L::Mul{<:ExpansionLayout,Lay}) where Lay<:AbstractLazyLayout = copy(Mul{ApplyLayout{typeof(*)},Lay}(L.A, L.B))

function broadcastbasis_layout(::typeof(+), _, _, a, b)
    try
        a ≠ b && error("Overload broadcastbasis(::typeof(+), ::$(typeof(a)), ::$(typeof(b)))")
    catch
        error("Overload broadcastbasis(::typeof(+), ::$(typeof(a)), ::$(typeof(b)))")
    end
    a
end

broadcastbasis_layout(::typeof(+), ::MappedBasisLayouts, ::MappedBasisLayouts, a, b) = broadcastbasis(+, demap(a), demap(b))[basismap(a), :]
function broadcastbasis_layout(::typeof(+), ::SubBasisLayout, ::SubBasisLayout, a, b)
    kr_a,jr_a = parentindices(a)
    kr_b,jr_b = parentindices(b)
    @assert kr_a == kr_b # frist axes must match
    view(broadcastbasis(+, parent(a), parent(b)), kr_a, union(jr_a,jr_b))
end
broadcastbasis_layout(::typeof(+), ::SubBasisLayout, _, a, b) = broadcastbasis(+, parent(a), b)
broadcastbasis_layout(::typeof(+), _, ::SubBasisLayout, a, b) = broadcastbasis(+, a, parent(b))

broadcastbasis(::typeof(+), a, b) = broadcastbasis_layout(+, MemoryLayout(a), MemoryLayout(b), a, b)
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


function equals_layout(::ExpansionLayout, ::ExpansionLayout, f, g)
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
sum_layout(::AbstractBasisLayout, Vm, dims) = error("Overload _sum(::$(typeof(Vm)), ::$(typeof(dims)))")

function sum_layout(::SubBasisLayout, Vm, dims)
    dims == 1 || error("not implemented")
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

###
# diff
###

diff_layout(::AbstractBasisLayout, Vm, dims...) = error("Overload diff(::$(typeof(Vm)))")

function diff_layout(::SubBasisLayout, Vm, dims::Integer=1)
    dims == 1 || error("not implemented")
    diff(parent(Vm); dims=dims)[:,parentindices(Vm)[2]]
end

function diff_layout(::WeightedBasisLayout{SubBasisLayout}, Vm, dims::Integer=1)
    dims == 1 || error("not implemented")
    w = weight(Vm)
    V = unweighted(Vm)
    view(diff(w .* parent(V)), parentindices(V)...)
end

function diff_layout(::MappedBasisLayouts, V, dims::Integer=1)
    kr = basismap(V)
    @assert kr isa AbstractAffineQuasiVector
    D = diff(demap(V); dims=dims)
    view(basis(D), kr, :) * (kr.A*coefficients(D))
end

diff_layout(::ExpansionLayout, A, dims...) = diff_layout(ApplyLayout{typeof(*)}(), A, dims...)


####
# Gram matrix
####

simplifiable(::Mul{<:AdjointBasisLayout, <:AbstractBasisLayout}) = Val(true)
function copy(M::Mul{<:AdjointBasisLayout, <:AbstractBasisLayout})
    A = (M.A)'
    A == M.B && return grammatrix(A)
    error("Not implemented")
end

grammatrix(A) = grammatrix_layout(MemoryLayout(A), A)
grammatrix_layout(_, A) = error("Not implemented")


function grammatrix_layout(::MappedBasisLayouts, P)
    Q = demap(P)
    kr = basismap(P)
    @assert kr isa AbstractAffineQuasiVector
    grammatrix(Q)/kr.A
end

weaklaplacian(A) = weaklaplacian_layout(MemoryLayout(A), A)
weaklaplacian_layout(_, A) = weaklaplacian_axis(axes(A,1), A)
weaklaplacian_axis(::Inclusion{<:Number}, A) = -(diff(A)'diff(A))

function copy(M::Mul{<:AdjointMappedBasisLayouts, <:MappedBasisLayouts})
    A = M.A'
    kr = basismap(A)
    @assert kr isa AbstractAffineQuasiVector
    @assert kr == basismap(M.B)
    demap(A)'demap(M.B) / kr.A
end



include("basisconcat.jl")
include("basiskron.jl")
include("splines.jl")
