struct KronBasisLayout <: AbstractBasisLayout end

QuasiArrays.kronlayout(::AbstractBasisLayout...) = KronBasisLayout()

"""
    KronExpansionLayout

is a MemoryLayout corresponding to a quasi-matrix corresponding to the 2D expansion K[x,y] == A[x]*X*B[y]'
"""
struct KronExpansionLayout{LayA, LayB} <: AbstractLazyLayout end
applylayout(::Type{typeof(*)}, ::LayA, ::CoefficientLayouts, ::AdjointBasisLayout{LayB}) where {LayA <: AbstractBasisLayout, LayB <: AbstractBasisLayout} = KronExpansionLayout{LayA,LayB}()