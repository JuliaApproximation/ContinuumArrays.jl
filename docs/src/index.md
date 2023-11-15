# ContinuumArrays.jl
A Julia package for working with bases as quasi-arrays

## The Basis interface

To add your own bases, subtype `Basis` and overload the following routines:

1. `axes(::MyBasis) = (Inclusion(mydomain), OneTo(mylength))`
2. `grid(::MyBasis, ::Integer)`
3. `getindex(::MyBasis, x, ::Integer)`
