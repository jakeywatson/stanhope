"""Phase C validation: learning curve across episodes (no hard_reset).

Runs 30 episodes per agent WITHOUT wiping learned priors between episodes,
then reports success rate across three sliding windows so any cross-episode
improvement shows up.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from scenarios.drone_search_v2 import DroneSearchV2Scenario  # noqa: E402


def run_learning_curve(agent_type: str, n_episodes: int = 30, seed: int = 12345) -> dict:
    import time
    np.random.seed(seed)
    scenario = DroneSearchV2Scenario()
    # Wipe learned priors at start so we see the curve from zero knowledge.
    scenario.hard_reset(agent_type)
    per_ep_success = []
    per_ep_score = []
    t_start = time.time()
    print(f"[{agent_type}] starting", flush=True)
    for ep in range(n_episodes):
        if ep > 0:
            scenario.reset(agent_type)  # soft reset — keep learned α
        summary = scenario.run_episode_summary()
        per_ep_success.append(summary['success'])
        per_ep_score.append(summary['reward'])
        if (ep + 1) % 5 == 0:
            print(f"  [{agent_type}] ep {ep+1}/{n_episodes}  "
                  f"elapsed {time.time()-t_start:.0f}s  "
                  f"last-steps {summary['steps']:.0f}  "
                  f"alpha_sum {scenario.world_alpha.sum():.0f}", flush=True)
    return {
        'agent': agent_type,
        'success': per_ep_success,
        'score': per_ep_score,
        'final_alpha': scenario.world_alpha.copy(),
    }


def window_rates(xs: list, window: int = 10) -> list:
    return [float(np.mean(xs[i:i + window])) for i in range(0, len(xs), window)]


def main():
    agents = ('greedy', 'active_inference', 'combined')
    results = {}
    for agent in agents:
        r = run_learning_curve(agent, n_episodes=30)
        results[agent] = r
        wins = window_rates(r['success'])
        score_w = window_rates(r['score'])
        print(f"\n{agent.upper():18s}")
        print(f"  success rate (eps 1-10, 11-20, 21-30): "
              f"{wins[0]*100:.1f}%  {wins[1]*100:.1f}%  {wins[2]*100:.1f}%")
        print(f"  mean score (eps 1-10, 11-20, 21-30):   "
              f"{score_w[0]:+.2f}  {score_w[1]:+.2f}  {score_w[2]:+.2f}")
        alpha = r['final_alpha']
        total = alpha.sum()
        frac = alpha / total
        print(f"  learned world α: empty={alpha[0]:.0f} bldg={alpha[1]:.0f} "
              f"decoy={alpha[2]:.0f} target={alpha[3]:.0f}  (sum={total:.0f})")
        print(f"  learned class rates: empty={frac[0]:.2%} bldg={frac[1]:.2%} "
              f"decoy={frac[2]:.2%} target={frac[3]:.2%}")

    print("\nPhase C acceptance: combined's success rate should grow across windows;")
    print("greedy is essentially flat (uses the prior only weakly).")


if __name__ == '__main__':
    main()
