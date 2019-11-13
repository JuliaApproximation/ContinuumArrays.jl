using ContinuumArrays, DifferentialEquations, Plots

##
# natural boundary conditions
##

L = LinearSpline(range(-1,1; length=20))
x = axes(L,1)
D = Derivative(x)
M = L'L
Δ = -((D*L)'D*L)
u0 = copy(L \ exp.(x))

heat(u,(M,Δ),t) = M\(Δ*u)
prob = ODEProblem(heat,u0,(0.0,1.0),(M,Δ))
sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8)

g = range(-1,1;length=1000)
@gif for t in 0.0:0.01:1.0
    plot(g, L[g,:]*sol(t); ylims=(0,3))
end

##
# Dirichlet boundary conditions
##

S = L[:,2:end-1]
M = S'S
Δ = -((D*S)'D*S)
u0 = (L \ broadcast(x -> (1-x^2)*exp(x), x))[2:end-1]
prob = ODEProblem(heat,u0,(0.0,1.0),(M,Δ))
sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8)

g = range(-1,1;length=1000)
@gif for t in 0.0:0.01:1.0
    plot(g, S[g,:]*sol(t); ylims=(0,3))
end