"""One-step Expected Free Energy for the retuned T-maze.

The initial decision is a replanning choice: go directly to an arm or inspect
the cue first. If the cue is selected, the agent replans on the next step with
updated beliefs rather than committing to a fixed arm in advance.
"""
import numpy as np
from numpy import ndarray
from math_utils import softmax, log_stable, entropy, kl_dirichlet, dirichlet_expected
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
) -> dict:
    """Evaluate the initial T-maze choices and select one."""
    log_pref = log_stable(softmax(c))
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
            efe = _two_step_cue_efe(
                A, B, d, a_conc, log_pref,
                w_extrinsic, w_salience, w_novelty,
            )
        G[pi] = efe['total']
        efe_details[pi] = efe

    if force_uniform:
        probs = np.full(N_POLICIES, 1.0 / N_POLICIES)
        chosen = int(np.random.choice(N_POLICIES))
    else:
        gamma = 1.0 / max(beta, 1e-8)
        probs = softmax(gamma * G)
        chosen = int(np.random.choice(N_POLICIES, p=probs))

    return {
        'efe': efe_details,
        'G': G.tolist(),
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
    log_pref = log_stable(softmax(c))
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
    if need_novelty and a_conc is not None and target_loc in (LOC_LEFT, LOC_RIGHT):
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

    # Step-1 extrinsic is dropped: in the paper's factored T-maze the cue lives
    # in a separate observation modality with neutral preferences, so visiting
    # the cue does not pay an outcome-preference cost. Salience and novelty of
    # the cue step still contribute.
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

    return {
        'extrinsic': extrinsic_2,
        'salience': step1['salience'] + salience_2,
        'novelty': step1['novelty'] + novelty_2,
        'total': float(
            w_extrinsic * extrinsic_2
            + w_salience * (step1['salience'] + salience_2)
            + w_novelty * (step1['novelty'] + novelty_2)
        ),
    }
