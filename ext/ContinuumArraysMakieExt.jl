module ContinuumArraysMakieExt

using ContinuumArrays
using ContinuumArrays: plotgridvalues, AbstractQuasiArray, _split_svec
import Makie: convert_arguments


function convert_arguments(g::AbstractQuasiArray)
    x,v = plotgridvalues(g)
    tuple(_split_svec(x)..., v)
end

println("hello")

end # module