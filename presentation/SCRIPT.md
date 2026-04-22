# Speaker script · Stanhope interview · condensed

~25 min · ~145 wpm · speak to the insight, let the slide carry the detail · keystone hits only

---

### Part 1 · Model (slides 1–6, ~5.5 min)

**1 · Title** (30s)
- Paper's big claim: curiosity doesn't need to be added as a bonus; it falls out of a single variational objective.
- Why it's true, where it bends, how it connects to drones.

**2 · Roadmap** (30s)
- Model · planning · evidence · Stanhope bridge.

**3 · T-maze task** (60s)
- Centre + safe arm (small, deterministic) + risky arm (large w.p. *p*, unknown).
- Deliberately boring. Only uncertainty is one scalar *p* — can't hide exploration in ε-greedy or entropy bonus.

**4 · Generative model** (90s)
- Joint *p(o, s, θ)* — you can sample from it.
- **One model, three jobs**: inference (invert), simulation (roll forward), meta-inference (reason about θ).
- Model-free RL has none of these → no place for curiosity to live.

**5 · Figure 2 · A, B, c, d** (60s)
- Discrete POMDP. Four pieces.
- Only **one column of A** is uncertain — the rest is deterministic. Whole learning problem is that column.
- A is matrix showing likelihood of observations given hidden states.
- B is transitions between hidden states. 
- C is prior expectations about observations - specifies agents preferences (reward structure).
- D is prior expectations about initial states (always start in the middle).
- γ is precision of policy selection — the lever for random vs directed exploration.



**6 · A-matrix · Dirichlet** (75s)
- Belief over risky column = Dir(α). α = [1,1] = max uncertainty.
- As counts accumulate, Beta sharpens. **Novelty** picks up that sharpening rate.

### Part 2 · Planning (slides 7–12, ~6.5 min)

**7 · Variational free energy** (90s)
- Exact posterior over trajectories = intractable sum.
- Minimise *F* subject to simplex → softmax update falls out of the Lagrangian (not chosen, derived).
- Forward-backward for HMM, in active-inference clothing.

**8 · Expected free energy** (90s)
- Can't minimise *F* on unseen futures → take expectation → *G*.
- Decomposes into **extrinsic − salience − novelty**, all with minus signs.
- Minimising *G* = low cost + high info gain. **Curiosity is not added; it falls out.**

**9 · What each term buys** (75s)
- Extrinsic = log-preferences scoring (the only term greedy RL sees).
- Salience = state info gain (≈0 in bare T-maze).
- Novelty = parameter info gain — **closed form via Dirichlet conjugacy**. Digammas at integer counts. No sampling.

**10 · Dirichlet conjugacy** (75s)
- Cat × Dir = Dir with counts incremented. Learning rule = α += 1 at observed index.
- E[θ] = α/α₀, E[ln θ] = ψ(αᵢ) − ψ(α₀).
- Second identity is what makes novelty exact — lose conjugacy and the derivation breaks.

**11 · Policy precision β** (60s)
- Policy posterior = softmax(−γG), γ ~ Γ(1, β).
- **β is the lever**: small β → commit; large β → random.
- Prior on γ lets the agent hedge when *G* values bunch — *"I genuinely don't know."*

**12 · Four things the paper doesn't tell you** (60s)
- Policy horizon τ = 2 (paper); I implemented full τ but kept 2 for replication.
- Analytic marginal *q(o|π)* with plug-in mean *Ā*.
- Clamp log-zero with ε (same as `spm_MDP_VB`).
- Novelty with *q(A)* vs plug-in — small diff when α₀ large. Happy to go deep.

### Part 3 · Evidence (slides 13–15, ~5.25 min)

**13 · T-maze sim** (120s incl. live demo)
- Python-in-browser via Pyodide. Same `.py` runs locally.
- Sliders: β = precision; ext/sal/nov = EFE weights. Dirichlet panel live.

> **Demo cues.** Default: T-Maze + Active Learning + Interactive.
> - `Step` once → α +1, novelty bar shrinks.
> - `Run 10` → novelty decays, agent settles (~15 trials).
> - Swap → `Greedy`, `Reset`, `Run 10` ×3 → commits to safe, never explores risky.
> - Swap → `Random`, `Reset`, `Run 10` ×2 → chaotic, both α grow evenly.
> - Optional: drag β slider for commit ↔ hedge.

**14 · Figure 6 · the key result** (60s)
- AL: P(risky) decays as uncertainty resolves → curiosity self-terminates.
- Random: flat — exploration not tied to learning.
- Greedy: stuck — unlucky prior = never recovers.
- Decay governed by ψ(α₀) → closed-form, you can *compute* when curiosity turns off. Replicates.

**15 · What I'd push back on** (75s)
- *A* assumes known structure — real perception must learn the column, not just its entries.
- Novelty is count-based — turns off before estimate reaches truth.
- Policy space is combinatorial — amortisation is where the field has gone since 2019.
- None fatal; these are the hinges.

### Part 4 · Extensions + bridge (slides 16–22, ~8 min)

**16 · Grid-maze** (60s)
- 10×10 + fog of war. *A* is a learned map.
- Salience now *matters* — visiting a cell reduces state entropy. Novelty still fires.
- τ = 2 fine; τ = 5 starts to feel the combinatorics.

**17 · Unknown site search (v2)** (60s)
- 15×15 random site, buildings occlude, targets + decoys, **no map**.
- 4-class cell belief + per-object target belief.
- **Transferable Dirichlet prior α** — learnt *across* episodes. Same digamma, richer hypothesis space.

> **Demo cues.** Scenario → `Unknown Site Search` (~2s build).
> - `Run 10` → drone surveys; heatmap saturates; ribbon cycles Explore/Scan/Confirm.
> - `Reset` (new layout, α persists) → `Train 10 eps` → α bars accumulate; mini curve fills.
> - `Reset α` to wipe if re-running cold.

**18 · Ablation** (75s)
- Zero *w_sal*, *w_nov*; clamp β = 0.125. 30 eps/agent.
- **Combined 83→47 (−37pp). AI 80→47 (−33pp).** Every smart agent collapses to the same extrinsic-only baseline — *that's* the point: curiosity is what distinguishes them.
- **Counter-example**: AL 27 → 47 (*anti-gap* +20pp). Novelty without salience is diffuse — agent roams. Forcing greedy commit-to-best rescues it by shutting roaming down.
- Controls: greedy 40, random 30 — flat under toggle.

> **Demo cues.** Mode → `Experimenter`.
> - Episodes = 10, ablation **off** → `Run Benchmark` (~60 s). Combined ~83, AI ~80.
> - `Clear Results` → flip ablation **on** → `Run Benchmark` → smart agents collapse to ~47%; controls unchanged.

**19 · What each gap tells us** (60s)
- Combined −37, AI −33: **salience does most of the per-episode lifting**. Novelty adds cross-episode transfer.
- AL anti-gap: curiosity must be **aimed**. Novelty = "what haven't I seen"; salience = "what resolves what I care about". Need both.
- One-liner: combined doubles greedy (83 vs 40).

**20 · Munin Dynamics** (75s) · Feb–Aug 2025, volunteer
- Problem: detect quadcopter @ 200m, 150 m/s closing, <30 mm board, $250 BoM.
- Built: camera/SBC selection (Johnson); FOMO, int8-quant, on-platform; capture-detect-track on Digi CC93.
- **90% recall** (FOMO) vs **mAP50 0.27** (YOLOv11n) same compute. ~200 KB model. <1 ms @ 128².
- Ceiling of that system is a better detector; I'm here for the planner above it.

**21 · Frazer-Nash** (75s) · Sept 2022 – Apr 2024
- Kargu-class multirotor, EO/IR, maritime. Tight perception→control loop on SWaP hardware.
- Rebuilt vision **20 → 70+ fps** with better accuracy. Led tech design on **£235k bid**.
- Broke under ambiguity — **confidence ≠ uncertainty**. Missing layer: shared uncertainty-aware plane between perception and control.

**22 · Where it meets Stanhope** (75s)
- Shipped twice: pixels → target in ms, no posterior, retrain-per-domain, greedy downstream.
- AI adds: explicit *q(θ)*; salience + novelty planned *ahead*; one objective for pursuit + exploration; self-terminating curiosity.
- Stanhope's Real World Model sits at this layer. Frazer-Nash SENTINEL (RL-for-sensor-mgmt) = same problem shape.
- **The offering**: perception pipeline *underneath* — twice. Here for the agent *on top*.

**23 · Thanks** (15s)
- Sim + source on slide. Questions.

---

### Runtime & trim

| Section | Slides | min |
|---|---|---|
| Open + roadmap | 1–2 | 1.0 |
| Model | 3–6 | 4.5 |
| Planning | 7–11 | 6.5 |
| Caveats + sim | 12–13 | 3.0 |
| Results | 14–15 | 2.25 |
| Extensions + ablation | 16–19 | 4.25 |
| Stanhope bridge | 20–22 | 3.75 |
| Close | 23 | 0.25 |
| **Total** | | **~25.5** |

**Cuts if short:** skip 12 (−1'), fold 16 into 17 (−45s), 20 s headline on 19 (−40s), drop ablation detail on 18 (−30s) → ~22 min.
**Expansions if long:** digamma identity on 10; extra sim pass on 13; harder landing on SENTINEL on 22.

### Live-demo pre-flight

- **Open `localhost:5173` ≥5 min before** — splash must say "Ready." and fade.
- **Throwaway `Step`** in each scenario — first step JITs the Python path.
- **Experimenter:** Episodes = 10, Batch = 5 for live runs. Full 30-ep benchmark takes 2–3 min — don't run that live.
- **Gotchas**:
  - `Reset` in Experimenter wipes the leaderboard (not a soft reset) — use `Clear Results` between ablation off/on.
  - `Step` relabels to `Run Batch` in Experimenter — it runs a batch, not one step.
  - Don't resize the window mid-demo (training-curve canvas doesn't reflow).
  - If Pyodide wedges after long idle, reload — 10 s recovery.
