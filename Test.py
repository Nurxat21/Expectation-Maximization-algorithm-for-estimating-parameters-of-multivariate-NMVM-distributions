import numpy as np
from EM import *
from scipy.stats import norm, gennorm
from scipy.special import kv
from numpy.linalg import cholesky
# -------- Parameters -------- #
n = 1000            # number of samples
d = 2               # dimension
lam = 1.0           # λ
chi = 1.0           # χ
psi = 1.0           # ψ
mu = np.array([0.0, 0.0])
gamma = np.array([0.2, -0.1])
Sigma = np.array([[1.0, 0.3],
                [0.3, 1.0]])
# Cholesky factor
A = cholesky(Sigma)

# -------- Sample W from GIG(λ, χ, ψ) -------- #
def sample_gig(lam, chi, psi, size=1):
    # SciPy has gengamma but not GIG; we simulate GIG via rejection
    # For now, we approximate with method of Devroye
    from scipy.stats import geninvgauss
    return geninvgauss.rvs(p=lam, b=np.sqrt(chi*psi), scale=1/psi, size=size)

W = sample_gig(lam, chi, psi, size=n)

# -------- Generate Z ∼ N(0, I) -------- #
Z = np.random.randn(n, d)

# -------- Construct samples from mGH -------- #
X = mu + W[:, None] * gamma + np.sqrt(W)[:, None] * (Z @ A.T)
em = EM_algorithm(X)
params, log_likelihood_GH= em.run("GH", True)
print("Estimated parameters for GH distribution:")
print("lambda: ", params[0])
print("a: ", params[1])
print("b: ", params[2])
print("mu: ", params[3])
print("Sigma: ", params[4])
print("gamma: ", params[5])
print("Final log-likelihood: ", log_likelihood_GH[-1])
