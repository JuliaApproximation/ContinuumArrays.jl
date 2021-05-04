using ContinuumArrays, Plots

L = LinearSpline(range(0,1; length=5))
plot(L)