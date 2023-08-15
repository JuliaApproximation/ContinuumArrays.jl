using ContinuumArrays, Test
import Makie
import Makie: Point

@testset "Plotting" begin
    L = LinearSpline(0:5)
    u = L*(2:7)

    fig = Makie.lines(u);
    @test fig.plot.positions[] == Point.(0:5, 2:7)
    fig = Makie.plot(u);
    @test fig.plot.positions[] == Point.(0:5, 2:7)
    fig = Makie.plot(L);
    @test fig.plot.P[] == L
end