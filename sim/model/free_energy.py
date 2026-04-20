"""Expected Free Energy computation (Equation 7, Schwartenbeck et al. 2019).

G(π) = Σ_τ [ extrinsic_value(τ) + ambiguity(τ) + novelty(τ) ]

Where:
  - extrinsic_value (risk): E_q[log P(o|C)] — do outcomes match preferences?
  - ambiguity (salience): E_q[H[P(o|s)]] — how ambiguous is state→observation mapping?
  - novelty (parameter info gain): expected KL divergence of Dirichlet posterior update
"""
import numpy as np
from numpy import ndarray
from math_utils import softmax, log_stable, entropy, kl_dirichlet, dirichlet_expected


def compute_efe(
    A: ndarray,
    c: ndarray,
    state_after_action: int,
    dirichlet_conc: ndarray,
    enable_extrinsic: bool = True,
    enable_salience: bool = True,
    enable_novelty: bool = True,
) -> dict:
    """Compute expected free energy for taking an action that leads to a given state.

    Args:
        A: Current expected A-matrix (4 obs × 3 states), using Dirichlet expectations
        c: Log-preference vector (4,) over observations
        state_after_action: The state the agent will be in after the action (1=safe, 2=risky)
        dirichlet_conc: Current Dirichlet concentrations for the risky arm [a_high, a_none]
        enable_*: Toggles for each EFE component (for ablation studies)

    Returns:
        dict with 'extrinsic', 'salience', 'novelty', 'total'
    """
    # Predicted observations given the state we'll end up in
    # P(o | s_after_action) from current A-matrix beliefs
    predicted_obs = A[:, state_after_action]  # (4,)

    # --- 1. Extrinsic value (risk) ---
    # E_q[log P(o|C)] where C encodes preferences
    # = Σ_o P(o|s) * log σ(c)_o
    extrinsic = 0.0
    if enable_extrinsic:
        log_pref = log_stable(softmax(c))
        extrinsic = float(predicted_obs @ log_pref)

    # --- 2. Ambiguity / Salience (hidden state info gain) ---
    # E_q[H[P(o|s)]] = entropy of the observation distribution for this state
    # High ambiguity = bad (agent prefers unambiguous states)
    salience = 0.0
    if enable_salience:
        salience = -entropy(predicted_obs)  # negative because lower entropy = better

    # --- 3. Novelty (parameter info gain) ---
    # Expected information gain about A-matrix parameters
    # = Expected KL divergence between posterior (after observing) and prior (current)
    novelty = 0.0
    if enable_novelty and state_after_action == 2:  # only risky arm has learnable params
        novelty = _compute_novelty(dirichlet_conc)

    total = extrinsic + salience + novelty

    return {
        'extrinsic': float(extrinsic),
        'salience': float(salience),
        'novelty': float(novelty),
        'total': float(total),
    }


def _compute_novelty(conc: ndarray) -> float:
    """Expected information gain from visiting the risky arm.

    For each possible observation o ∈ {high_reward, no_reward}:
      - Compute P(o) from current Dirichlet expectation
      - Compute posterior Dirichlet after observing o
      - Compute KL[posterior || prior]
      - Weight by P(o)

    Returns the expected KL divergence (always positive — more novelty = more info gain).
    """
    p = dirichlet_expected(conc)  # [p_high, p_none]

    expected_kl = 0.0
    for i in range(len(conc)):
        # Posterior after observing outcome i
        posterior = conc.copy()
        posterior[i] += 1.0

        kl = kl_dirichlet(posterior, conc)
        expected_kl += p[i] * kl

    return expected_kl
