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
2. `diff(::MyBasis, dims=1)` to support differentiation and `Derivative`. 
3. `grammatrix(::MyBasis)` to support `Q'Q`. 
4. `ContinuumArrays._sum(::MyBasis, dims)` and `ContinuumArrays._cumsum(::MyBasis, dims)` to support definite and indefinite integeration.
5. `plotgrid(::MyBasis, n...)`: return `n` grid points suitable for plotting the basis. The default value for `n` is 10,000. 


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
transform
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


## Interal Routines

```@docs
ContinuumArrays.TransformFactorization
```

```@docs
ContinuumArrays.AbstractConcatBasis
```
```@docs
ContinuumArrays.MulPlan
```
```@docs
ContinuumArrays.PiecewiseBasis
```
```@docs
ContinuumArrays.Map
```
```@docs
ContinuumArrays.MappedFactorization
```
```@docs
ContinuumArrays.basis
```
```@docs
ContinuumArrays.InvPlan
```
```@docs
ContinuumArrays.VcatBasis
```
```@docs
ContinuumArrays.WeightedFactorization
```
```@docs
ContinuumArrays.HvcatBasis
```
```@docs
ContinuumArrays.ProjectionFactorization
```
```
