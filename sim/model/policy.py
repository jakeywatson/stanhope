"""Policy evaluation and selection via Expected Free Energy.

Policies are scored by their (negative) EFE, then selected via softmax
with precision parameter γ = 1/β.
"""
import numpy as np
from numpy import ndarray
from math_utils import softmax
from free_energy import compute_efe


def evaluate_policies(
    A: ndarray,
    c: ndarray,
    dirichlet_conc: ndarray,
    beta: float = 1.0,
    enable_extrinsic: bool = True,
    enable_salience: bool = True,
    enable_novelty: bool = True,
) -> dict:
    """Evaluate both policies (go_safe, go_risky) and return scores + selection probabilities.

    Args:
        A: Expected A-matrix (4×3)
        c: Log-preference vector (4,)
        dirichlet_conc: Dirichlet concentrations for risky arm [a_high, a_none]
        beta: Precision parameter (higher = more random; γ = 1/β)
        enable_*: EFE component toggles

    Returns:
        dict with 'efe_safe', 'efe_risky' (each has extrinsic/salience/novelty/total),
        'policy_probs' {safe, risky}, 'chosen_action' (int)
    """
    # Compute EFE for each action
    efe_safe = compute_efe(
        A, c, state_after_action=1,
        dirichlet_conc=dirichlet_conc,
        enable_extrinsic=enable_extrinsic,
        enable_salience=enable_salience,
        enable_novelty=enable_novelty,
    )
    efe_risky = compute_efe(
        A, c, state_after_action=2,
        dirichlet_conc=dirichlet_conc,
        enable_extrinsic=enable_extrinsic,
        enable_salience=enable_salience,
        enable_novelty=enable_novelty,
    )

    # Policy selection: π = softmax(γ · G) where G = ext + sal + nov (utility to maximize)
    gamma = 1.0 / max(beta, 1e-8)
    G = np.array([efe_safe['total'], efe_risky['total']])
    probs = softmax(gamma * G)

    # Sample action
    action = int(np.random.choice(2, p=probs))

    return {
        'efe_safe': efe_safe,
        'efe_risky': efe_risky,
        'policy_probs': {'safe': float(probs[0]), 'risky': float(probs[1])},
        'chosen_action': action,
    }
