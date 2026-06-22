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
end