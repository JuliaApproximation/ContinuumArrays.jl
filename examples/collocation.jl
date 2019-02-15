using ContinuumArrays, FillArrays, InfiniteArrays
import ContinuumArrays.QuasiArrays: Inclusion, QuasiDiagonal

P = Legendre()
X = QuasiDiagonal(Inclusion(-1..1))

@test X[-1:0.1:1,-1:0.1:1] == Diagonal(-1:0.1:1)

axes(X)
J = pinv(P)*X*P

J - I
Vcat(Hcat(1, Zeros(1,∞)), J)
