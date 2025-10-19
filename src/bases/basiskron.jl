struct KronBasisLayout <: AbstractBasisLayout end

QuasiArrays.kronlayout(::AbstractBasisLayout...) = KronBasisLayout()

"""
    KronExpansionLayout

is a MemoryLayout corresponding to a quasi-matrix corresponding to the 2D expansion K[x,y] == A[x]*X*B[y]'
"""
struct KronExpansionLayout{LayA, LayB} <: AbstractLazyLayout end
applylayout(::Type{typeof(*)}, ::LayA, ::CoefficientLayouts, ::AdjointBasisLayout{LayB}) where {LayA <: AbstractBasisLayout, LayB <: AbstractBasisLayout} = KronExpansionLayout{LayA,LayB}()

sublayout(::KronExpansionLayout, inds) = sublayout(ApplyLayout{typeof(*)}(), inds)
sublayout(::KronExpansionLayout{LayA, LayB}, inds::Type{<:Tuple{Inclusion,AbstractVector}}) where {LayA,LayB} = ExpansionLayout{LayA}()


sub_basis_layout(::KronExpansionLayout, P, j) = first(arguments(P))


function sub_coefficients_layout(::KronExpansionLayout, P, j)
   _,X,Bt = arguments(P)
    X * Bt[:,j]
end

sum_layout(::KronExpansionLayout, F, dims...) = sum_layout(ApplyLayout{typeof(*)}(), F, dims...)