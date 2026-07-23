"""
Engine: projective F/R-symbol representation of the braid group B_3 on the
d-dimensional fusion space of 3 anyons of spin j in SU(2)_k, plus the
M-tilde witness-bound engine.

Building blocks:
- q-integer [n] = sin(n pi/(k+2))/sin(pi/(k+2)); q-factorial; q-6j (Racah formula).
- R-symbol R^{jj}_c = (-1)^{2j-c} exp(i pi (h_c - 2h_j)), h_l = l(l+1)/(k+2).
- F-matrix (recoupling (12)3 <-> 1(23)): [F]_{ef} = (-1)^{3j+J} sqrt([2e+1][2f+1]) {j j e; j J f}_q.
- Braid generators: sigma1 = diag(R^{jj}_e); sigma2 = F diag(R^{jj}_f) F^dag.
- M-tilde(B) = Gram matrix of sym(2 Phi_B - Phi_{B^2}) in an orthonormal Hermitian
  basis (d^2 x d^2); lambda_max(M-tilde) = witness bound (max over Q on {Tr Q^2 = d}).
  For d=2 this is tight and reproduces the M(U) construction.
Labels are DOUBLED (a = 2j) where needed, so half-integrality stays exact.
"""
import numpy as np
from functools import lru_cache

# ---------- q-Zahlen ----------
def q_int(n, k):
    """[n]_q = sin(n pi/(k+2))/sin(pi/(k+2)), n integer >=0."""
    if n == 0:
        return 0.0
    return np.sin(n*np.pi/(k+2.0))/np.sin(np.pi/(k+2.0))

@lru_cache(maxsize=None)
def q_fact(n, k):
    """[n]_q! = prod_{m=1}^n [m]_q ; [0]! = 1."""
    if n < 0:
        return 0.0
    r = 1.0
    for m in range(1, n+1):
        r *= q_int(m, k)
    return r

# ---------- admissibility (DOUBLED labels a=2j) ----------
def allowed(a, b, c, k):
    """SU(2)_k: is fusion j1 x j2 -> j3 allowed? a=2j1, b=2j2, c=2j3."""
    if (a + b + c) % 2:
        return False
    if c < abs(a - b) or c > a + b:
        return False
    if c > 2*k - a - b:
        return False
    return 0 <= c <= 2*k

# ---------- q-6j (spin arguments, half-integers allowed; integer triads internally) ----------
def _tri(a, b, c, k):
    """Triangle coefficient Delta(a,b,c) (spins). 0 if not admissible (truncated)."""
    A = [a+b-c, a-b+c, -a+b+c]
    if any(abs(x - round(x)) > 1e-9 or x < -1e-9 for x in A):
        return 0.0
    x1, x2, x3 = (int(round(x)) for x in A)
    den = int(round(a+b+c+1))
    # truncated admissibility: 2j-labels via 2*(...)
    if not allowed(int(round(2*a)), int(round(2*b)), int(round(2*c)), k):
        return 0.0
    num = q_fact(x1, k)*q_fact(x2, k)*q_fact(x3, k)
    if abs(q_fact(den, k)) < 1e-15:
        return 0.0
    return np.sqrt(max(num, 0.0)/q_fact(den, k))

def sixj(a, b, e, c, d, f, k):
    """Quantum 6j {a b e; c d f}_q (spin arguments)."""
    tri = _tri(a, b, e, k)*_tri(c, d, e, k)*_tri(a, c, f, k)*_tri(b, d, f, k)
    if tri == 0.0:
        return 0.0
    t = [a+b+e, c+d+e, a+c+f, b+d+f]
    s = [a+b+c+d, a+d+e+f, b+c+e+f]
    zlo = int(np.ceil(max(t) - 1e-9)); zhi = int(np.floor(min(s) + 1e-9))
    S = 0.0
    for z in range(zlo, zhi+1):
        terms = [z-t[0], z-t[1], z-t[2], z-t[3], s[0]-z, s[1]-z, s[2]-z]
        if any(x < 0 for x in [int(round(y)) for y in terms]):
            continue
        den = 1.0
        ok = True
        for y in terms:
            qf = q_fact(int(round(y)), k)
            if abs(qf) < 1e-15:
                ok = False; break
            den *= qf
        if not ok:
            continue
        S += ((-1)**z) * q_fact(z+1, k) / den
    return tri * S

# ---------- conformal weights + R-symbol ----------
def h_spin(j, k):
    return j*(j+1.0)/(k+2.0)

def R_jj(j, c, k):
    """R^{jj}_c = (-1)^{2j-c} exp(i pi (h_c - 2 h_j))."""
    sign = (-1.0)**int(round(2*j - c))
    return sign*np.exp(1j*np.pi*(h_spin(c, k) - 2*h_spin(j, k)))

# ---------- sector + braid generators ----------
def sector_charges(k, twoj, twoJ):
    """Intermediate charges 2e (=2c) in the total-charge-J sector. d = len."""
    out = []
    for twoe in range(0, 2*k+1):
        if allowed(twoj, twoj, twoe, k) and allowed(twoe, twoj, twoJ, k):
            out.append(twoe)
    return out

def F_matrix(k, twoj, twoJ, charges):
    """[F]_{ef} = (-1)^{3j+J} sqrt([2e+1][2f+1]) {j j e; j J f}_q. Real orthogonal."""
    j = twoj/2.0; J = twoJ/2.0
    d = len(charges)
    F = np.zeros((d, d))
    glob = (-1.0)**int(round(3*j + J))
    for ie, twoe in enumerate(charges):
        e = twoe/2.0
        for jf, twof in enumerate(charges):
            f = twof/2.0
            val = glob*np.sqrt(q_int(twoe+1, k)*q_int(twof+1, k))*sixj(j, j, e, j, J, f, k)
            F[ie, jf] = val
    return F

def braid_generators(k, twoj, twoJ):
    """sigma1, sigma2 (d x d) in the (12)3 basis. sigma1 = diag(R^{jj}_e); sigma2 = F diag(R^{jj}_f) F^dag."""
    charges = sector_charges(k, twoj, twoJ)
    d = len(charges)
    if d < 2:
        return None
    j = twoj/2.0
    Rdiag = np.array([R_jj(j, twoe/2.0, k) for twoe in charges], dtype=complex)
    s1 = np.diag(Rdiag)
    F = F_matrix(k, twoj, twoJ, charges)
    s2 = F @ np.diag(Rdiag) @ F.conj().T
    return dict(s1=s1, s2=s2, F=F, charges=charges, d=d)

# ---------- to_su(d) (remove det phase -> projective) ----------
def to_sud(U):
    d = U.shape[0]
    det = np.linalg.det(U)
    return U / det**(1.0/d)

# ---------- orthonormal Hermitian basis (d^2) ----------
def herm_basis(d):
    B = [np.eye(d, dtype=complex)/np.sqrt(d)]
    for a in range(d):
        for b in range(a+1, d):
            S = np.zeros((d, d), complex); S[a, b]=1; S[b, a]=1; B.append(S/np.sqrt(2))
            A = np.zeros((d, d), complex); A[a, b]=-1j; A[b, a]=1j; B.append(A/np.sqrt(2))
    for m in range(1, d):  # diagonal traceless (Gell-Mann)
        diag = np.zeros(d); diag[:m]=1; diag[m]=-m
        diag = diag/np.sqrt(m*(m+1))
        B.append(np.diag(diag).astype(complex))
    return B  # len d^2

# ---------- M-tilde Gram matrix + witness bound ----------
def Mtilde_lambda_max(B, basis):
    """lambda_max(M-tilde(B)) = max over {Tr Q^2 = d} K3(Q). M-tilde = sym(2 Phi_B - Phi_{B^2}), Gram in orthonormal basis."""
    B2 = B @ B; Bd = B.conj().T; B2d = B2.conj().T
    n = len(basis)
    G = np.empty((n, n))
    BEBd = [B @ E @ Bd for E in basis]
    B2EB2d = [B2 @ E @ B2d for E in basis]
    for i, Ei in enumerate(basis):
        for jj in range(n):
            t1 = np.trace(Ei @ BEBd[jj]).real
            t2 = np.trace(Ei @ B2EB2d[jj]).real
            G[i, jj] = 2*t1 - t2
    G = (G + G.T)/2
    return float(np.linalg.eigvalsh(G)[-1])

def K3_Qfixed(B, Q):
    """K3(B,Q) = 2C(B)-C(B²), C(U)=(1/d)Re Tr[Q U Q U†]."""
    d = B.shape[0]
    def C(U):
        return (1.0/d)*np.real(np.trace(Q @ U @ Q @ U.conj().T))
    return 2*C(B) - C(B @ B)

# ---------- projective group enumeration ----------
def enumerate_group(s1, s2, tol=1e-8, max_size=4000):
    Id = np.eye(s1.shape[0], dtype=complex)
    gens = [to_sud(s1), to_sud(s1.conj().T), to_sud(s2), to_sud(s2.conj().T)]
    elems = [Id.copy()]
    def seen(U):
        return any(np.max(np.abs(U - V)) < tol for V in elems)
    frontier = [Id.copy()]
    for g in gens:
        if not seen(g):
            elems.append(g); frontier.append(g)
    while frontier:
        nf = []
        for U in frontier:
            for g in gens:
                W = to_sud(U @ g)
                if not seen(W):
                    elems.append(W); nf.append(W)
                    if len(elems) > max_size:
                        return None
        frontier = nf
    return elems
