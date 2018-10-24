module ContinuumArrays
include("axisarrays/AbstractAxisArrays.jl")
using .AbstractAxisArrays


####
# Interval indexing support
####

using IntervalSets
import .AbstractAxisArrays: _length, checkindex, Adjoint, Transpose
import Base: @_inline_meta

struct ℵ₀ <: Number end
_length(::AbstractInterval) = ℵ₀

checkindex(::Type{Bool}, inds::AbstractInterval, i::Real) = (leftendpoint(inds) <= i) & (i <= rightendpoint(inds))
function checkindex(::Type{Bool}, inds::AbstractInterval, I::AbstractArray)
    @_inline_meta
    b = true
    for i in I
        b &= checkindex(Bool, inds, i)
    end
    b
end

end
