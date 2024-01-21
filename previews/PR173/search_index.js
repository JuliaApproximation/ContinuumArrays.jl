var documenterSearchIndex = {"docs":
[{"location":"#ContinuumArrays.jl","page":"Home","title":"ContinuumArrays.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia package for working with bases as quasi-arrays","category":"page"},{"location":"#The-Basis-interface","page":"Home","title":"The Basis interface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To add your own bases, subtype Basis and overload the following routines:","category":"page"},{"location":"","page":"Home","title":"Home","text":"axes(::MyBasis) = (Inclusion(mydomain), OneTo(mylength))\ngrid(::MyBasis, ::Integer)\ngetindex(::MyBasis, x, ::Integer)\n==(::MyBasis, ::MyBasis)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Optional overloads are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"plan_transform(::MyBasis, sizes::NTuple{N,Int}, region) to plan a transform, for a tensor","category":"page"},{"location":"","page":"Home","title":"Home","text":"product of the specified sizes, similar to plan_fft.","category":"page"},{"location":"","page":"Home","title":"Home","text":"diff(::MyBasis, dims=1) to support differentiation and Derivative. \ngrammatrix(::MyBasis) to support Q'Q. \nContinuumArrays._sum(::MyBasis, dims) and ContinuumArrays._cumsum(::MyBasis, dims) to support definite and indefinite integeration.","category":"page"},{"location":"#Routines","page":"Home","title":"Routines","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Derivative","category":"page"},{"location":"#ContinuumArrays.Derivative","page":"Home","title":"ContinuumArrays.Derivative","text":"Derivative(axis)\n\nrepresents the differentiation (or finite-differences) operator on the specified axis.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"transform","category":"page"},{"location":"#ContinuumArrays.transform","page":"Home","title":"ContinuumArrays.transform","text":"transform(A, f)\n\nfinds the coefficients of a function f expanded in a basis defined as the columns of a quasi matrix A. It is equivalent to\n\nA \\ f.(axes(A,1))\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"expand","category":"page"},{"location":"#ContinuumArrays.expand","page":"Home","title":"ContinuumArrays.expand","text":"expand(A, f)\n\nexpands a function f im a basis defined as the columns of a quasi matrix A. It is equivalent to\n\nA / A \\ f.(axes(A,1))\n\n\n\n\n\nexpand(v)\n\nfinds a natural basis for a quasi-vector and expands in that basis.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"grid","category":"page"},{"location":"#ContinuumArrays.grid","page":"Home","title":"ContinuumArrays.grid","text":"grid(P, n...)\n\nCreates a grid of points. if n is unspecified it will be sufficient number of points to determine size(P,2) coefficients. If n is an integer or Block its enough points to determine n coefficients. If n is a tuple then it returns a tuple of grids corresponding to a tensor-product. That is, a 5⨱6 2D transform would be\n\n(x,y) = grid(P, (5,6))\nplan_transform(P, (5,6)) * f.(x, y')\n\nand a 5×6×7 3D transform would be\n\n(x,y,z) = grid(P, (5,6,7))\nplan_transform(P, (5,6,7)) * f.(x, y', reshape(z,1,1,:))\n\n\n\n\n\n","category":"function"},{"location":"#Interal-Routines","page":"Home","title":"Interal Routines","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.TransformFactorization","category":"page"},{"location":"#ContinuumArrays.TransformFactorization","page":"Home","title":"ContinuumArrays.TransformFactorization","text":"TransformFactorization(grid, plan)\n\nassociates a planned transform with a grid. That is, if F is a TransformFactorization, then F \\ f is equivalent to F.plan * f[F.grid].\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.AbstractConcatBasis","category":"page"},{"location":"#ContinuumArrays.AbstractConcatBasis","page":"Home","title":"ContinuumArrays.AbstractConcatBasis","text":"AbstractConcatBasis\n\nis an abstract type representing a block diagonal basis but with modified axes.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.MulPlan","category":"page"},{"location":"#ContinuumArrays.MulPlan","page":"Home","title":"ContinuumArrays.MulPlan","text":"MulPlan(matrix, dims)\n\nTakes a matrix and supports it applied to different dimensions.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.PiecewiseBasis","category":"page"},{"location":"#ContinuumArrays.PiecewiseBasis","page":"Home","title":"ContinuumArrays.PiecewiseBasis","text":"PiecewiseBasis(args...)\n\nis an analogue of Basis that takes the union of the first axis, and the second axis is a blocked concatenatation of args. If there is overlap, it uses the first in order.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.basis","category":"page"},{"location":"#ContinuumArrays.basis","page":"Home","title":"ContinuumArrays.basis","text":"basis(v)\n\ngives a basis for expanding given quasi-vector.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.InvPlan","category":"page"},{"location":"#ContinuumArrays.InvPlan","page":"Home","title":"ContinuumArrays.InvPlan","text":"InvPlan(factorization, dims)\n\nTakes a factorization and supports it applied to different dimensions.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.VcatBasis","category":"page"},{"location":"#ContinuumArrays.VcatBasis","page":"Home","title":"ContinuumArrays.VcatBasis","text":"VcatBasis\n\nis an analogue of Basis that vcats the values.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.WeightedFactorization","category":"page"},{"location":"#ContinuumArrays.WeightedFactorization","page":"Home","title":"ContinuumArrays.WeightedFactorization","text":"WeightedFactorization(w, F)\n\nweights a factorization F by w.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.HvcatBasis","category":"page"},{"location":"#ContinuumArrays.HvcatBasis","page":"Home","title":"ContinuumArrays.HvcatBasis","text":"VcatBasis\n\nis an analogue of Basis that hvcats the values, so they are matrix valued.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"ContinuumArrays.ProjectionFactorization","category":"page"},{"location":"#ContinuumArrays.ProjectionFactorization","page":"Home","title":"ContinuumArrays.ProjectionFactorization","text":"ProjectionFactorization(F, inds)\n\nprojects a factorization to a subset of coefficients. That is, if P is a ProjectionFactorization then P \\ f is equivalent to (F \\ f)[inds]\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"```","category":"page"}]
}
