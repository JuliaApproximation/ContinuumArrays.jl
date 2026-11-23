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

@testset "@simplify body line numbers" begin
    # line numbers of the body are kept, even when @simplify is called from the file it is defined in
    file = Symbol(pathof(ContinuumArrays) |> dirname |> d -> joinpath(d, "operators.jl"))
    body = Expr(:block, LineNumberNode(2, file), :(x + 1))
    ex = Expr(:block, LineNumberNode(100, file), Expr(:function, :(f(x)), body))
    source = LineNumberNode(1, file)
    ContinuumArrays._callsite_linenums!(ex, source, body)
    @test ex.args[1] ≡ source
    @test body.args[1] == LineNumberNode(2, file)
end
