"""Mathematical utilities for active inference computations.

Implements softmax, Dirichlet distribution functions, KL divergence,
and the digamma/gammaln functions needed for expected free energy.
"""
import numpy as np
from numpy import ndarray


def softmax(x: ndarray) -> ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def log_stable(x: ndarray) -> ndarray:
    """Log with floor to avoid -inf."""
    return np.log(np.maximum(x, 1e-16))


def dirichlet_expected(alpha: ndarray) -> ndarray:
    """Expected value of Dirichlet distribution: E[θ_i] = α_i / Σα."""
    return alpha / alpha.sum()


def dirichlet_entropy(alpha: ndarray) -> float:
    """Entropy of a Dirichlet distribution.

    H[Dir(α)] = log B(α) + (α₀ - K)ψ(α₀) - Σ(α_i - 1)ψ(α_i)
    where α₀ = Σα_i, K = len(α), ψ = digamma.
    """
    from scipy.special import gammaln, digamma as _digamma
    a0 = alpha.sum()
    K = len(alpha)
    log_B = gammaln(alpha).sum() - gammaln(a0)
    return log_B + (a0 - K) * _digamma(a0) - ((alpha - 1) * _digamma(alpha)).sum()


def kl_dirichlet(alpha: ndarray, beta: ndarray) -> float:
    """KL divergence between two Dirichlet distributions.

    D_KL[Dir(α) || Dir(β)] = log[B(β)/B(α)] + Σ(α_i - β_i)[ψ(α_i) - ψ(α₀)]
    """
    from scipy.special import gammaln, digamma as _digamma
    a0 = alpha.sum()
    b0 = beta.sum()
    log_B_ratio = (gammaln(beta).sum() - gammaln(b0)) - (gammaln(alpha).sum() - gammaln(a0))
    return log_B_ratio + ((alpha - beta) * (_digamma(alpha) - _digamma(a0))).sum()


def entropy(p: ndarray) -> float:
    """Shannon entropy H[p] = -Σ p_i log p_i."""
    p = np.maximum(p, 1e-16)
    return -(p * np.log(p)).sum()
