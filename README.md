# ContinuumArrays.jl
A package for representing quasi arrays with continuous dimensions

[![Build Status](https://travis-ci.org/JuliaApproximation/ContinuumArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaApproximation/ContinuumArrays.jl)
[![codecov](https://codecov.io/gh/JuliaApproximation/ContinuumArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/ContinuumArrays.jl)


A quasi array as implemented in [QuasiArrays.jl](https://github.com/JuliaApproximation/QuasiArrays.jl) is a 
generalization of an array that allows non-integer indexing via general axes. This package adds support for
infinite-dimensional axes, including continuous intervals. Thus it plays the same role as [InfiniteArrays.jl](https://github.com/JuliaArrays/InfiniteArrays.jl) does for standard arrays but now for quasi arrays. 

A simple example is the identity function on the interval `0..1`. This can be created using `Inclusion(d)`,
which returns `x` if `x in d` is true, otherwise throws an error:
```julia
julia> using ContinuumArrays, IntervalSets

julia> x = Inclusion(0..1.0)
Inclusion(0.0..1.0)

julia> size(x) # uncountable (aleph-1)
(ℵ₁,)

julia> axes(x) # axis is itself
(Inclusion(0.0..1.0),)

julia> x[0.1] # returns the input
0.1

julia> x[1.1] # throws an error
ERROR: BoundsError: attempt to access Inclusion(0.0..1.0)
  at index [1.1]
Stacktrace:
 [1] throw_boundserror(::Inclusion{Float64,Interval{:closed,:closed,Float64}}, ::Tuple{Float64}) at ./abstractarray.jl:538
 [2] checkbounds at /Users/solver/Projects/QuasiArrays.jl/src/abstractquasiarray.jl:287 [inlined]
 [3] getindex(::Inclusion{Float64,Interval{:closed,:closed,Float64}}, ::Float64) at /Users/solver/Projects/QuasiArrays.jl/src/indices.jl:158
 [4] top-level scope at REPL[14]:1
```

An important usage is representing bases and function approximation, and this package contains
a basic implementation of linear splines and heaviside functions. For example, we can construct splines
with evenly spaced nodes via:
```julia
julia> L = LinearSpline(0:0.2:1);

julia> size(L) # uncountable (alepha-1) by 11
(ℵ₁, 6)

julia> axes(L) # The interval 0.0..1.0 by 1:6. 
(Inclusion(0.0..1.0), Base.OneTo(6))

julia> L[[0.15,0.25,0.45],1:6] # can index like an array
3×6 Array{Float64,2}:
 0.25  0.75  0.0   0.0   0.0  0.0
 0.0   0.75  0.25  0.0   0.0  0.0
 0.0   0.0   0.75  0.25  0.0  0.0
```
Functions in this basis are represented by a lazy multiplication by a basis
and a vector of coefficients:
```julia
julia> f = L*[1,2,3,4,5,6]
QuasiArrays.ApplyQuasiArray{Float64,1,typeof(*),Tuple{Spline{1,Float64},Array{Int64,1}}}(*, (Spline{1,Float64}([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), [1, 2, 3, 4, 5, 6]))

julia> axes(f)
(Inclusion(0.0..1.0),)

julia> f[0.1]
1.5
```

Creating a finite element method is possible using standard array terminology. 
We always take the Lebesgue inner product associated with an axes, so in this
case the mass matrix is just `L'L`. Combined with a derivative operator allows
us to form the weak Laplacian.
```julia
julia> B = L[:,2:end-1]; # drop boundary terms to impose zero Dirichlet

julia> Δ = (D*B)'D*B # weak Laplacian
4×4 BandedMatrices.BandedMatrix{Float64,Array{Float64,2},Base.OneTo{Int64}}:
 10.0  -5.0    ⋅     ⋅ 
 -5.0  10.0  -5.0    ⋅ 
   ⋅   -5.0  10.0  -5.0
   ⋅     ⋅   -5.0  10.0

julia> B'f # right-hand side
4-element Array{Float64,1}:
 0.4
 0.6
 0.8
 1.0

 julia> c = Δ \ B'f # coefficients of Poisson
4-element Array{Float64,1}:
 0.24               
 0.4                
 0.43999999999999995
 0.3199999999999999 

julia> u = B*c; # expand in basis

julia> u[0.1] # evaluate at 0.1
0.12
```


