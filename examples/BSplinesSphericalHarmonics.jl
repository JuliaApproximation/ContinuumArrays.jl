SH = SphericalHarmonics()
CS = CubicSplines(0.0:0.01:1)
QS = QuadraticSplines(0.0:0.01:1)
LS = LinearSplines(0.0:0.01:1)
S = SH ⊗ CS

λ = @. -(1:∞)*((1:∞) +1)
L = BlockDiagonal(Diagonal.(Fill.(λ,1:2:∞))) # create a Block matrix with growing block sizes
Δ_s = Mul(SH, L, inv(SH)) # the Laplace–Boltrami operator
D_r = Mul(QS, SymTridiagonal(...), inv(CS))
R = Mul(QS, Banded(...), inv(QS)) # Diagonal(Spl.points)
D̃_r = Mul(LS, SymTridiagonal(...), inv(QS))

Δ = I ⊗ (D̃_r*R^2*D_r) + Δ_s ⊗ I # creates Laplacian as a Mul(S, ..., inv(S)) operator

# finite_dimensional case
N = 100
S_N = Spl * SH[:, Block.(1:N)]  # take only first N degree spherical harmonics

Δ_N = Δ*S_N  # Knows that inv(S)*S_N === inv(S_N)

# R²Δ_N is a lazy BandedBlockBandedMatrix, with diagonal blockbandwidths
backend = BandedBlockBandedMatrix{Float64,MPIMatrix{Float64}}(undef, size(R²Δ_N), (0,0), bandwidths(D_r);
                        workers = ...) # distribute different blocks based on workers

MPI_Δ_N =  Mul(S_N, backend, inv(S_N))

MPI_Δ_N .= Δ_N # populates MPI array, generating data on workers remotely


f = LazyFun((x,y,z) -> cos(x*y)+z, S_N)  # not sure how constructors fit in yet...
MPI_f = Mul(S_N, BlockVector{Float64,MPIVector{Float64}}(undef, blocklengths(backend, 2))

MPI_f .= f # automatically generates data remotely, via MPI version of FastTransforms

MPI_v = similar(MPI_f)

C = Mul(SH ⊗ QS, I ⊗ (QS' * LS), inv(SH ⊗ LS))
MPI_C = ...

MPI_v .= Mul(MPI_Δ_N, MPI_f) # applies the Laplacian and fills in missing information.
