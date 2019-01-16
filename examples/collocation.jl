using ContinuumArrays
import ContinuumArrays.QuasiArrays: Inclusion, QuasiDiagonal
P = Legendre()
axes(P)

X = QuasiDiagonal(Inclusion(axes(P,1)))



axes(X)
pinv(P)*X*P

X*P
QuasiDiagonal(axes(P,1))


@which Base.Slice(Base.OneTo(5)) == Base.OneTo(5)
