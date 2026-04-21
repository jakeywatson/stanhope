"""Phase A smoke test for drone_search_v2.

Runs N episodes with the greedy agent and reports score distribution.
Acceptance: greedy average score ≤ 0 over 30 episodes (see spec §11 Phase A).
"""
import sys
import os
import numpy as np

# Make `model/` importable (matches the layout used by the Pyodide bridge).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from scenarios.drone_search_v2 import DroneSearchV2Scenario  # noqa: E402


def run_batch(agent_type: str, n_episodes: int = 30, seed: int = 12345) -> dict:
    np.random.seed(seed)
    scenario = DroneSearchV2Scenario()
    scores = []
    successes = 0
    tp_total = fp_total = tn_total = fn_total = 0
    steps_total = 0
    discovered_total = 0
    objs_total = 0
    for ep in range(n_episodes):
        scenario.hard_reset(agent_type)
        summary = scenario.run_episode_summary()
        scores.append(summary['reward'])
        successes += int(summary['success'])
        extras = summary['extras']
        tp_total += extras['tp']; fp_total += extras['fp']
        tn_total += extras['tn']; fn_total += extras['fn']
        steps_total += summary['steps']
        discovered_total += extras['discovered']
        objs_total += extras['n_true_targets']
    return {
        'agent': agent_type,
        'mean_score': float(np.mean(scores)),
        'median_score': float(np.median(scores)),
        'min_score': float(np.min(scores)),
        'max_score': float(np.max(scores)),
        'success_rate': successes / n_episodes,
        'tp': tp_total, 'fp': fp_total, 'tn': tn_total, 'fn': fn_total,
        'avg_steps': steps_total / n_episodes,
        'avg_discovered': discovered_total / n_episodes,
        'avg_n_targets': objs_total / n_episodes,
    }


def main():
    for agent in ('random', 'greedy', 'active_inference', 'active_learning', 'combined'):
        r = run_batch(agent, n_episodes=30)
        print(f"\n{r['agent'].upper():18s}  n=30")
        print(f"  score  mean={r['mean_score']:+.2f}  median={r['median_score']:+.2f}  "
              f"min={r['min_score']:+.2f}  max={r['max_score']:+.2f}")
        print(f"  success rate: {r['success_rate']*100:.1f}%")
        print(f"  tp={int(r['tp'])}  fp={int(r['fp'])}  "
              f"tn={int(r['tn'])}  fn={int(r['fn'])}")
        print(f"  avg steps: {r['avg_steps']:.1f}")
        print(f"  avg discovered: {r['avg_discovered']:.2f} / {r['avg_n_targets']:.2f} true targets")

    print("\nPhase B acceptance: combined ≥80%, greedy ≤30-45%, "
          "ablations (active_inference/active_learning) in between.")


if __name__ == '__main__':
    main()
