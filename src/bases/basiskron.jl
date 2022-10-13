struct KronBasisLayout <: AbstractBasisLayout end

QuasiArrays.kronlayout(::AbstractBasisLayout...) = KronBasisLayout()