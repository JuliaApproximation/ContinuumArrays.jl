# ContinuumArrays.jl
A Julia package for working with bases as quasi-arrays

## The Basis interface

To add your own bases, subtype `Basis` and overload the following routines:

1. `axes(::MyBasis) = (Inclusion(mydomain), OneTo(mylength))`
2. `grid(::MyBasis, ::Integer)`
3. `getindex(::MyBasis, x, ::Integer)`
4. `==(::MyBasis, ::MyBasis)`

Optional overloads are:

1. `plan_transform(::MyBasis, sizes::NTuple{N,Int}, region)` to plan a transform, for a tensor
product of the specified sizes, similar to `plan_fft`.
2. `plan_itransform(::MyBasis, sizes::NTuple{N,Int}, region)` to plan an inverse transform, similar to `plan_ifft`.
The default multiplies by the basis evaluated on the grid.
3. `diff(::MyBasis, dims=1)` to support differentiation and `Derivative`. 
4. `grammatrix(::MyBasis)` to support `Q'Q`. 
5. `ContinuumArrays._sum(::MyBasis, dims)` and `ContinuumArrays._cumsum(::MyBasis, dims)` to support definite and indefinite integeration.
6. `plotgrid(::MyBasis, n...)`: return `n` grid points suitable for plotting the basis. The default value for `n` is 10,000. 


## Two-dimensional transforms

For a function of two variables, `P \ F / Q'` gives the coefficient matrix in the bases
`P` and `Q`. Its rows correspond to basis functions in the first variable and its columns
to basis functions in the second.

```julia
P = LinearSpline(range(0, 1, 5))
F = [1+x+y+x*y for x in 0..1, y in 0..1]
C = P \ F / P'
(P*C*P')[0.1, 0.2] ≈ 1.32
```

## Differential Operators


```@docs
Derivative
```

```@docs
Laplacian
```

```@docs
AbsLaplacian
```

```@docs
laplacian
```

```@docs
abslaplacian
```

```@docs
weaklaplacian
```

## Routines

```@docs
affine
```
```@docs
transform
```
```@docs
plan_transform
```
```@docs
itransform
```
```@docs
plan_itransform
```
```@docs
expand
```
```@docs
grid
```
```@docs 
plotgrid
```
```@docs
⊎
```


## Interal Routines

```@docs
ContinuumArrays.TransformFactorization
```
```@docs
ContinuumArrays.AbstractConcatBasis
```
```@docs
ContinuumArrays.basis
```
```@docs
ContinuumArrays.expandsexactly
```
```@docs
ContinuumArrays.exactexpansion
```
```@docs
ContinuumArrays.HvcatBasis
```
```@docs
ContinuumArrays.InvPlan
```
```@docs
ContinuumArrays.KronExpansionLayout
```
```@docs
ContinuumArrays.Map
```
```@docs
ContinuumArrays.MappedFactorization
```
```@docs
ContinuumArrays.MulPlan
```
```@docs
ContinuumArrays.PiecewiseBasis
```
```@docs
ContinuumArrays.ProjectionFactorization
```
```@docs
ContinuumArrays.uplus_components
```
```@docs
ContinuumArrays.uplus_size
```
```@docs
ContinuumArrays.VcatBasis
```
```@docs
ContinuumArrays.WeightedFactorization
```

