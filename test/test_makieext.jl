using ContinuumArrays, Test
using Makie
using Makie: Point

@testset "Plotting" begin
    L = LinearSpline(0:5)
    u = L*(2:7)

    fig = Makie.lines(u);
    @test fig.plot.positions[] == Point.(0:5, 2:7)
    fig = Makie.plot(u);
    @test fig.plot.positions[] == Point.(0:5, 2:7)
    fig = Makie.plot(L);
    @test fig.plot.P[] == L

    L = LinearSpline(range(0,1; length=4))
    C = reshape(Vector(1:16), 4, 4)
    F = L * C * L'

    fig = Makie.plot(F);
    @test fig.plot isa Contourf

    fig = Makie.plot(DiracDelta(0.5, 0..1));
    @test fig.plot.positions[] == Point.([0.5, 0.5], [0.0, 1.0])

    fig = Makie.plot(Spline{-1}(0:3) * [1,2]);
    @test isequal(fig.plot.positions[], Point.([1,1,1,2,2,2], [0,1,NaN,0,2,NaN]))
end
