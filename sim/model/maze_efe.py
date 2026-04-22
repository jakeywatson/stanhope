"""One-step Expected Free Energy for the retuned T-maze.

The initial decision is a replanning choice: go directly to an arm or inspect
the cue first. If the cue is selected, the agent replans on the next step with
updated beliefs rather than committing to a fixed arm in advance.
"""
import numpy as np
from numpy import ndarray
from math_utils import softmax, entropy, kl_dirichlet, dirichlet_expected
from maze_model import (
    N_CONTEXTS,
    N_STATES,
    N_POLICIES,
    POLICY_NAMES,
    POLICY_LEFT_DIRECT,
    POLICY_RIGHT_DIRECT,
    LOC_CUE,
    LOC_LEFT,
    LOC_RIGHT,
)


def evaluate_maze_policies(
    A: ndarray,
    B: ndarray,
    c: ndarray,
    d: ndarray,
    a_conc: ndarray | None = None,
    beta: float = 1.0,
    w_extrinsic: float = 1.0,
    w_salience: float = 1.0,
    w_novelty: float = 1.0,
    force_uniform: bool = False,
    no_cue: bool = False,
) -> dict:
    """Evaluate the initial T-maze choices and select one.

    If ``no_cue`` is True, the cue-then-best policy is disabled (Fig 3/6 task
    variant, where cue observation is not available). Its EFE is set to -inf
    and probability forced to zero.
    """
    # c is already the log-preference (Schwartenbeck et al. 2019, eq. 4).
    # Taking log(softmax(c)) introduces a per-step log-normaliser that does
    # not cancel cleanly when comparing policies of different horizons, so
    # the multi-step cue-then-best path was being penalised just for having
    # two steps. Using c directly preserves policy-invariant shifts.
    log_pref = c
    G = np.zeros(N_POLICIES)
    efe_details = {}
    need_salience = w_salience > 0
    need_novelty = w_novelty > 0

    for pi, policy_name in enumerate(POLICY_NAMES):
        if policy_name == POLICY_LEFT_DIRECT:
            efe = _with_total(
                _one_step_efe(A, B, d, LOC_LEFT, a_conc, log_pref,
                              need_salience, need_novelty),
                w_extrinsic,
                w_salience,
                w_novelty,
            )
        elif policy_name == POLICY_RIGHT_DIRECT:
            efe = _with_total(
                _one_step_efe(A, B, d, LOC_RIGHT, a_conc, log_pref,
                              need_salience, need_novelty),
                w_extrinsic,
                w_salience,
                w_novelty,
            )
        else:
            if no_cue:
                # Disabled policy — use -inf for G-level selection but 0 for
                # UI-visible totals so the policy bar simply reads 0 in the
                # panel instead of -∞.
                efe = {'extrinsic': 0.0, 'salience': 0.0, 'novelty': 0.0,
                       'total': 0.0, '_disabled': True}
            else:
                efe = _two_step_cue_efe(
                    A, B, d, a_conc, log_pref,
                    w_extrinsic, w_salience, w_novelty,
                )
        G[pi] = efe['total'] if not efe.get('_disabled') else float('-inf')
        efe_details[pi] = efe

    # Select among only the active policies. In no_cue mode the cue policy's
    # G is -inf and should never be chosen; also zero its probability for UI.
    active_mask = np.isfinite(G)
    if force_uniform:
        probs = np.where(active_mask, 1.0 / max(int(active_mask.sum()), 1), 0.0)
        active_indices = np.where(active_mask)[0]
        chosen = int(np.random.choice(active_indices))
    else:
        gamma = 1.0 / max(beta, 1e-8)
        safe_G = np.where(active_mask, G, -1e9)
        probs = softmax(gamma * safe_G)
        probs = np.where(active_mask, probs, 0.0)
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        chosen = int(np.random.choice(N_POLICIES, p=probs))

    # Sanitize -inf for JSON serialization (the browser gets this via
    # pyodide → JSON, which cannot encode Infinity).
    G_out = [float(v) if np.isfinite(v) else None for v in G]
    for k, efe in efe_details.items():
        efe.pop('_disabled', None)
    return {
        'efe': efe_details,
        'G': G_out,
        'probs': probs.tolist(),
        'chosen_policy': chosen,
    }


def select_best_arm_target(
    A: ndarray,
    B: ndarray,
    c: ndarray,
    belief: ndarray,
    a_conc: ndarray | None,
    w_extrinsic: float,
    w_salience: float,
    w_novelty: float,
    force_uniform: bool = False,
) -> tuple[int, dict]:
    """Choose the best arm after an observation has updated the belief state."""
    # c is already the log-preference (Schwartenbeck et al. 2019, eq. 4).
    # Taking log(softmax(c)) introduces a per-step log-normaliser that does
    # not cancel cleanly when comparing policies of different horizons, so
    # the multi-step cue-then-best path was being penalised just for having
    # two steps. Using c directly preserves policy-invariant shifts.
    log_pref = c
    need_salience = w_salience > 0
    need_novelty = w_novelty > 0
    left = _with_total(
        _one_step_efe(A, B, belief, LOC_LEFT, a_conc, log_pref,
                      need_salience, need_novelty),
        w_extrinsic,
        w_salience,
        w_novelty,
    )
    right = _with_total(
        _one_step_efe(A, B, belief, LOC_RIGHT, a_conc, log_pref,
                      need_salience, need_novelty),
        w_extrinsic,
        w_salience,
        w_novelty,
    )
    if force_uniform:
        if int(np.random.choice(2)) == 1:
            return LOC_RIGHT, right
        return LOC_LEFT, left
    if right['total'] > left['total']:
        return LOC_RIGHT, right
    return LOC_LEFT, left


def _one_step_efe(
    A: ndarray,
    B: ndarray,
    belief: ndarray,
    target_loc: int,
    a_conc: ndarray | None,
    log_pref: ndarray,
    need_salience: bool = True,
    need_novelty: bool = True,
) -> dict:
    """Compute one-step EFE components for moving to a specific location.

    Salience and novelty are only computed when their EFE weight is non-zero —
    the component consumes ~half the benchmark runtime for active-learning
    agents, and greedy/random gain nothing by evaluating it.
    """
    next_belief = B[:, :, target_loc] @ belief
    next_belief = _normalize(next_belief)

    predicted_obs = A @ next_belief
    predicted_obs = _normalize(predicted_obs)

    extrinsic = float(predicted_obs @ log_pref)

    salience = 0.0
    if need_salience:
        ambiguity = 0.0
        for state in range(N_STATES):
            if next_belief[state] > 1e-16:
                ambiguity += next_belief[state] * entropy(A[:, state])
        salience = entropy(predicted_obs) - ambiguity

    novelty = 0.0
    # Only the RIGHT (risky) arm has learnable reliabilities — LEFT is
    # deterministic and the cue is known. Cap novelty evaluation accordingly.
    if need_novelty and a_conc is not None and target_loc == LOC_RIGHT:
        arm_conc = a_conc[target_loc]
        if getattr(arm_conc, 'ndim', 1) == 1:
            novelty = _param_info_gain(arm_conc)
        else:
            state_base = target_loc * N_CONTEXTS
            for context in range(N_CONTEXTS):
                novelty += next_belief[state_base + context] * _param_info_gain(arm_conc[context])

    return {
        'extrinsic': float(extrinsic),
        'salience': float(salience),
        'novelty': float(novelty),
        'next_belief': next_belief,
        'predicted_obs': predicted_obs,
    }


def _with_total(efe: dict, w_extrinsic: float, w_salience: float, w_novelty: float) -> dict:
    """Attach a weighted total utility to an EFE component breakdown."""
    return {
        'extrinsic': efe['extrinsic'],
        'salience': efe['salience'],
        'novelty': efe['novelty'],
        'total': float(
            w_extrinsic * efe['extrinsic']
            + w_salience * efe['salience']
            + w_novelty * efe['novelty']
        ),
    }


def _normalize(values: ndarray) -> ndarray:
    """Normalize a probability vector while guarding against numerical underflow."""
    clipped = np.maximum(values, 1e-16)
    total = clipped.sum()
    if total <= 1e-16:
        return np.full_like(clipped, 1.0 / clipped.size)
    return clipped / total


def _param_info_gain(conc: ndarray) -> float:
    """Expected info gain about reward probabilities for a given arm."""
    p = dirichlet_expected(conc)

    expected_kl = 0.0
    for i in range(len(conc)):
        posterior = conc.copy()
        posterior[i] += 1.0
        expected_kl += p[i] * kl_dirichlet(posterior, conc)

    return expected_kl


def _two_step_cue_efe(
    A: ndarray,
    B: ndarray,
    belief: ndarray,
    a_conc: ndarray | None,
    log_pref: ndarray,
    w_extrinsic: float,
    w_salience: float,
    w_novelty: float,
) -> dict:
    """EFE of (visit cue → take best arm given the cue observation).

    The paper scores multi-step policies by summing per-step EFE. We do that
    here by enumerating the cue's possible observations, updating the belief,
    selecting the better arm under the updated belief, and weighting the
    follow-up EFE by the predicted observation probability.
    """
    need_salience = w_salience > 0
    need_novelty = w_novelty > 0
    step1 = _one_step_efe(A, B, belief, LOC_CUE, a_conc, log_pref,
                          need_salience, need_novelty)

    predicted_obs = step1['predicted_obs']
    cue_belief = step1['next_belief']

    # Count the cue-visit's own extrinsic (small and near-zero under the
    # paper's neutral cue preferences, but non-zero — dropping it created a
    # tie between left-direct and cue-then-best that greedy broke 50/50).
    extrinsic_2 = 0.0
    salience_2 = 0.0
    novelty_2 = 0.0

    for obs_idx, p_obs in enumerate(predicted_obs):
        if p_obs <= 1e-12:
            continue
        likelihood = A[obs_idx, :]
        unnorm = likelihood * cue_belief
        total = unnorm.sum()
        if total <= 1e-16:
            continue
        post_belief = unnorm / total

        left = _one_step_efe(A, B, post_belief, LOC_LEFT, a_conc, log_pref,
                             need_salience, need_novelty)
        right = _one_step_efe(A, B, post_belief, LOC_RIGHT, a_conc, log_pref,
                              need_salience, need_novelty)
        left_total = _with_total(left, w_extrinsic, w_salience, w_novelty)['total']
        right_total = _with_total(right, w_extrinsic, w_salience, w_novelty)['total']
        chosen = right if right_total > left_total else left

        extrinsic_2 += float(p_obs) * chosen['extrinsic']
        salience_2 += float(p_obs) * chosen['salience']
        novelty_2 += float(p_obs) * chosen['novelty']

    extrinsic_total = step1['extrinsic'] + extrinsic_2
    salience_total = step1['salience'] + salience_2
    novelty_total = step1['novelty'] + novelty_2
    return {
        'extrinsic': extrinsic_total,
        'salience': salience_total,
        'novelty': novelty_total,
        'total': float(
            w_extrinsic * extrinsic_total
            + w_salience * salience_total
            + w_novelty * novelty_total
        ),
    }
