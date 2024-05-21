module ContinuumArraysMakieExt

using ContinuumArrays, Makie
using ContinuumArrays: plotgridvalues, AbstractQuasiArray, AbstractQuasiMatrix, AbstractQuasiVector, _split_svec
import Makie: convert_arguments, plot!


function Makie.convert_arguments(p::PointBased, g::AbstractQuasiVector)
    x,v = plotgridvalues(g)
    convert_arguments(p, _split_svec(x)..., v)
end


function Makie.convert_arguments(p::GridBased, g::AbstractQuasiVector)
    x,v = plotgridvalues(g)
    convert_arguments(p, _split_svec(x)..., v)
end



@recipe(QuasiPlot, P) do scene
    Theme(
    )
end

Makie.plottype(a::AbstractQuasiVector) = Lines
Makie.plottype(a::AbstractQuasiMatrix) = QuasiPlot

function Makie.plot!(sc::QuasiPlot)
    x,v = plotgridvalues(sc[:P][])
    x2 = _split_svec(x)
    for j in axes(v,2)
        lines!(sc, x, v[:,j])
    end
    sc
end




end # module