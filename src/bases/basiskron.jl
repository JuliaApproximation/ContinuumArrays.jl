struct KronBasisLayout <: AbstractBasisLayout end

QuasiArrays.kronlayout(::AbstractBasisLayout...) = KronBasisLayout()

function transform_rows(B::QuasiAdjoint{<:Any,<:ApplyQuasiMatrix{<:Any,typeof(\)}}, g)
    P, F = arguments(parent(B))
    MemoryLayout(P) isa AbstractBasisLayout || return B[g, :]
    # The rows of (P \ F)' are adjoints of column transforms at the selected points.
    transform_ldiv(P, view(F, :, g))'
end

"""
    KronExpansionLayout

is a MemoryLayout corresponding to a quasi-matrix corresponding to the 2D expansion K[x,y] == A[x]*X*B[y]'
"""
struct KronExpansionLayout{LayA, LayB} <: AbstractLazyLayout end
applylayout(::Type{typeof(*)}, ::LayA, ::CoefficientLayouts, ::AdjointBasisLayout{LayB}) where {LayA <: AbstractBasisLayout, LayB <: AbstractBasisLayout} = KronExpansionLayout{LayA,LayB}()

sublayout(::KronExpansionLayout, inds) = sublayout(ApplyLayout{typeof(*)}(), inds)
sublayout(lay::KronExpansionLayout, ::Type{<:Tuple{Inclusion,Inclusion}}) = lay
sublayout(::KronExpansionLayout{LayA, LayB}, inds::Type{<:Tuple{Inclusion,AbstractVector}}) where {LayA,LayB} = ExpansionLayout{LayA}()

function arguments(::KronExpansionLayout, V::SubQuasiArray)
    P, C, Qt = arguments(parent(V))
    x, y = parentindices(V)
    view(P, x, :), C, view(Qt', y, :)'
end
expand_layout(::KronExpansionLayout, v) = v
function basis_layout(::KronExpansionLayout, v)
    P, _, Qt = arguments(v)
    P, Qt'
end
coefficients_layout(::KronExpansionLayout, v) = arguments(v)[2]
sub_basis_layout(::KronExpansionLayout, V) = first(arguments(parent(V)))
sub_basis_layout(lay::KronExpansionLayout, V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{Inclusion,Inclusion}}) = basis_layout(lay, V)
sub_coefficients_layout(::KronExpansionLayout, P, j::Inclusion) = coefficients(P)


function sub_coefficients_layout(::KronExpansionLayout, P, j)
   _,X,Bt = arguments(P)
    X * Bt[:,j]
end

sum_layout(::KronExpansionLayout, F, dims...) = sum_layout(ApplyLayout{typeof(*)}(), F, dims...)

diff_layout(::KronExpansionLayout, F, order...; dims...) = diff_layout(ApplyLayout{typeof(*)}(), F, order...; dims...)

copy(L::Ldiv{Lay,<:KronExpansionLayout}) where Lay<:AbstractBasisLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))


function plotgrid_layout(::KronExpansionLayout, P)
    A,X,Bt = arguments(P)
    plotgrid(A,max(20, maximum(colsupport(X)))), plotgrid(parent(Bt), max(20, maximum(rowsupport(X))))
end


for op in (:real, :imag)
    @eval function layout_broadcasted(::Tuple{KronExpansionLayout}, ::typeof($op), f)
        A,X,Bt = arguments(f)
        @assert isreal(A) && isreal(Bt)
        real(A) * $op(X) * real(Bt)
    end
end


LazyArrays._mul_arguments(lay::KronExpansionLayout, A) = arguments(lay, A)

laplacian_layout(::KronExpansionLayout, F, order=1; dims...) = diff(F, 2order; dims=1) + diff(F,2order; dims=2)
