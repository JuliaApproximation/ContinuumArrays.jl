using ContinuumArrays, QuasiArrays, Test
import ContinuumArrays: mul, ldiv, simplifiable

struct SimplifyLineA <: QuasiArrays.LazyQuasiMatrix{Float64} end
struct SimplifyLineB <: QuasiArrays.LazyQuasiMatrix{Float64} end

const SIMPLIFY_MUL_LINE = @__LINE__() + 1
ContinuumArrays.@simplify *(A::SimplifyLineA, B::SimplifyLineB) = 1
const SIMPLIFY_LDIV_LINE = @__LINE__() + 1
ContinuumArrays.@simplify \(A::SimplifyLineA, B::SimplifyLineB) = 2

@testset "@simplify line numbers" begin
    # generated methods refer to the call site so that coverage and stack traces are attributed there
    for (f, line) in ((mul, SIMPLIFY_MUL_LINE), (ldiv, SIMPLIFY_LDIV_LINE))
        m = which(f, Tuple{SimplifyLineA,SimplifyLineB})
        @test m.file == Symbol(@__FILE__)
        @test m.line == line
    end
    m = which(simplifiable, Tuple{typeof(*),SimplifyLineA,SimplifyLineB})
    @test m.file == Symbol(@__FILE__)
    @test m.line == SIMPLIFY_MUL_LINE
end
