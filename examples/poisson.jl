using ContinuumArrays, Plots

###
# Dirichlet
####

L = LinearSpline(range(0,1; length=10_000))[:,2:end-1]
D = Derivative(L)
Δ = -((D*L)'D*L)
M = L'L
f = expand(L, exp)
u = L * (Δ \ (M*f))
plot(u)

# we can also use cholesky
ũ = L * (cholesky(Symmetric(-Δ)) \ -(M*f))
plot(ũ)