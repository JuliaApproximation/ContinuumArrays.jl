using ContinuumArrays, Test
import RecipesBase

@testset "Plotting" begin
    L = LinearSpline(0:5)
    rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), L)
    @test rep[1].args == (L.points,L[L.points,:])

    rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), L[:,1:3])
    @test rep[1].args == (L.points,L[L.points,1:3])

    @test plotgrid(L[:,1:3],3) == grid(L[:,1:3]) == grid(L[:,1:3],3) == L.points
    

    u = L*randn(6)
    rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
    @test rep[1].args == (L.points,u[L.points])

    @testset "padded" begin
        u = L * Vcat(rand(3), Zeros(3))
        rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
        @test rep[1].args == (L.points,u[L.points])
    end

    @testset "Chebyshev and weighted Chebyshev" begin
        T =  Chebyshev(10)
        w =  ChebyshevWeight()
        wT = w .* T
        x =  axes(T, 1)
    
        u = T * Vcat(rand(3), Zeros(7))
        v = wT * Vcat(rand(3), Zeros(7))
    
        rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
        @test rep[1].args == (grid(T), u[grid(T)])
        wrep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), v)
        @test wrep[1].args[1] == grid(wT)
        @test wrep[1].args[2] ≈ v[grid(wT)]
    
        @test plotgrid(v) == plotgrid(u) == grid(T) == grid(wT) == plotgrid_layout(MemoryLayout(v), v) == plotgrid_layout(MemoryLayout(u), u)
        y = affine(0..1, x)
        @test plotgrid(T[y,:]) == (plotgrid(T) .+ 1)/2
    end

    @testset "basiskron" begin
        F = L*randn(6,6)*L'     
        rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), F)
        @test rep[1].args == (0:5, 0:5, F.args[2])
    end
end