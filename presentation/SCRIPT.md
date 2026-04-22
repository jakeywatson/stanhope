# Speaker script · Stanhope interview · condensed

~25 min · ~145 wpm · speak to the insight, let the slide carry the detail · keystone hits only

---

### Part 1 · Model (slides 1–6, ~5.5 min)

**1 · Title** (30s)
- The paper's big claim is that curiosity doesn't need to be bolted on as an exploration bonus — it falls out naturally from a single variational objective.
- I'll walk through why that claim holds, where I think it bends, and how the same machinery shows up in drone perception work I've been doing.

**2 · Roadmap** (30s)
- Four parts: the generative model, the planning objective, live evidence in the sim, and how this connects to what Stanhope is building.

**3 · T-maze task** (60s)
- Three-state maze: centre, a safe arm that pays a small deterministic reward, and a risky arm that pays large with probability *p* — and *p* is what the agent has to learn.
- The task is deliberately boring. The only uncertainty in the whole world is that one scalar *p*, which means you can't hide any exploration behaviour inside ε-greedy or an entropy bonus — whatever curiosity you see has to come straight from the objective.

**4 · Generative model** (90s)
- The agent carries a full joint *p(o, s, θ)* over observations, hidden states, and parameters — not just a policy. That means you can actually sample from it.
- **One model, three jobs**: invert it to do inference, roll it forward to simulate futures, and reason *about* θ itself for meta-inference — which is where curiosity lives.
- Model-free RL has none of those three capabilities — no forward model, no explicit θ — so there's literally no place in the architecture for curiosity to live as anything other than a reward-shaping hack.

**5 · Figure 2 · A, B, c, d** (60s)
- Under the hood it's a discrete POMDP, built from four pieces plus one precision scalar.
- Key point for later: only **one column of A** is actually uncertain — the rest is deterministic. The whole learning problem collapses into that single column.
- **A** is the likelihood matrix — probability of each observation given each hidden state.
- **B** is the transition matrix between hidden states under each action.
- **c** is the prior over observations — this is how you specify preferences / reward structure, as desired observations.
- **d** is the prior over initial states — here, a point mass on "centre".
- **γ** is precision of policy selection — it's the lever that dials between random exploration and commitment to the best policy.

**6 · A-matrix · Dirichlet** (75s)
- Belief over the risky column is a Dirichlet, Dir(α). Setting α = [1,1] means a uniform belief — maximum uncertainty over *p*.
- Every time you pull the risky arm, the corresponding count goes up by one. The Dirichlet sharpens, and **novelty** — which I'll define properly in a moment — is exactly the rate at which that sharpening is expected to happen.
- This is what the paper means by "learning A": not retraining a network, but Bayesian updating of a conjugate prior over the likelihood column, in closed form.


### Part 2 · Planning (slides 7–12, ~6.5 min)

**7 · Variational free energy** (90s)
- Exact posterior inference over full state trajectories is an intractable sum — same reason classical HMMs need forward-backward.
- The trick: minimise free energy *F* over a variational posterior subject to the simplex constraint. The softmax belief update isn't chosen — it falls directly out of the Lagrangian.
- What you end up with is forward-backward for an HMM, in active-inference clothing. Same algorithm, different framing.

**8 · Expected free energy** (90s)
- You can't minimise *F* on data you haven't observed yet. So for planning, you take the expectation over predicted future observations — that gives you **expected free energy** *G*.
- Once you expand it out, *G* decomposes into three terms with minus signs: **extrinsic − salience − novelty**. Preference + two info-gain terms.
- Minimising *G* therefore means *simultaneously* low pragmatic cost and high epistemic information gain. **Curiosity isn't added on — it falls out of the same objective that drives reward-seeking.**

**9 · What each term buys** (75s)
- **Extrinsic** is just scoring predicted observations against log-preferences *c*. It's the only term a greedy RL agent would see.
- **Salience** is expected information gain about hidden *states* — resolving confusion about where you are. In the bare T-maze this is roughly zero, because state is observed; it only lights up once you have a cue (Fig 8) or fog of war (grid maze).
- **Novelty** is expected information gain about *parameters* — how much the Dirichlet over A would sharpen. Closed-form thanks to conjugacy: digammas evaluated at integer counts, no sampling, no Monte Carlo.

**10 · Dirichlet conjugacy** (75s)
- Categorical likelihood times Dirichlet prior is another Dirichlet with counts incremented. So the learning rule is literally "α += 1 at the index you observed" — one line of code.
- Two identities do the heavy lifting: the mean **E[θ] = α/α₀** and, more importantly, **E[ln θ] = ψ(αᵢ) − ψ(α₀)** — the expected log-probability.
- That second identity is what lets novelty be evaluated in closed form. The moment you lose conjugacy — say, with a neural likelihood — this whole derivation breaks and you're back to sampling.

**11 · Policy precision β** (60s)
- The policy posterior is a softmax of negative expected free energy: q(π) = σ(−γG), with γ itself drawn from Gamma(1, β).
- **β is the lever**: small β sharpens γ, so the agent commits to the best policy; large β flattens the posterior towards uniform random.
- The Gamma prior on γ also lets the agent *modulate its own confidence* — when the G-values for different policies bunch together, γ shrinks naturally. It's the model saying *"I genuinely don't know which one is better, so don't commit."*

**12 · Four things the paper doesn't tell you** (60s)
- **Policy horizon.** Paper fixes τ = 2 (one step lookahead). I implemented general τ but stuck with τ = 2 for replication — anything longer starts to feel the combinatorics.
- **Expected observation marginal.** I use the analytic *q(o|π) = A · q(s|π)* with plug-in mean *Ā*, which avoids sampling noise at the cost of assuming the A-matrix is at its mean.
- **Numerical safety.** *ln 0* is a real risk with sparse A — I clamp it with a small ε, which is exactly what the reference `spm_MDP_VB` SPM code does.
- **Novelty with q(A) vs plug-in.** The strictly-correct form takes an expectation over the full Dirichlet q(A); the plug-in version uses the mean. Difference is small when α₀ is large, noticeable when it's small. Happy to go deep if anyone asks.

### Part 3 · Evidence (slides 13–15, ~5.25 min)

**13 · T-maze sim** (120s incl. live demo)
- Full Python (NumPy) model running in the browser via Pyodide. The same `.py` files run locally for validation — no JavaScript reimplementation.
- Sliders map directly to the maths: **β** is the policy precision, **ext/sal/nov** are the three EFE weights. The Dirichlet concentration panel on the right updates live as observations arrive.
- Three T-maze variants in the scenario dropdown — one per task from the paper:
  - **Volatile Context** (Fig 8A) — context resamples every trial, so the cue stays informative forever. Cue usage should be flat.
  - **Stable Context** (Fig 8B/8C) — context is drawn once per experiment. The d-vector concentrates as cue observations accumulate, and cue usage collapses. This is the iconic Fig 8C drop.
  - **Active Learning — no cue** (Fig 3/6C) — the cue pathway is disabled; only the Dirichlet over the risky arm's reward probability is learnable. The only way to gain information is to actually pull the risky arm.

> **Demo cues.** Default: T-Maze (Volatile) + Active Learning + Interactive.
> - `Step` once → α +1, novelty bar shrinks.
> - `Run 10` → novelty decays, agent settles (~15 trials).
> - Swap → `Greedy`, `Reset`, `Run 10` ×3 → commits to safe, never explores risky.
> - Swap → `Random`, `Reset`, `Run 10` ×2 → chaotic, both α grow evenly.
> - Optional: drag β slider for commit ↔ hedge.

**14 · Figure 6 + Figure 8 · the key results** (75s)
- **Fig 6C (Active Learning, no cue)** — Active Learning agent: probability of pulling the risky arm decays from ~1.0 to ~0.5 as the Dirichlet over *p* concentrates. Curiosity self-terminates. Random agent stays flat — exploration not tied to learning. Greedy gets stuck if the initial estimate is unlucky. The decay rate is governed by ψ(α₀), so you can *compute* when curiosity turns off in closed form.
- **Fig 8C (Active Inference, stable context)** — Active Inference agent: probability of visiting the cue collapses from ~1.0 to ~0.1 over 32 trials, as the d-vector over contexts concentrates on the true one. Salience info gain vanishes, and "cue-then-best" loses to the direct-arm policy.
- **Volatile context (Fig 8A)** — same agent, but now the context resamples every trial. P(cue visit) stays flat around 0.7 because the cue is always informative — there's nothing to learn across trials.
- In the Experimenter view, the new **"P(cue visit) by trial"** and **"P(risky visit) by trial"** charts plot all four shapes side-by-side, directly comparable with the paper figures.

> **Demo cues.** Mode → `Experimenter`. Episodes = 30, Batch = 10.
> - Scenario → `Stable Context` → `Run Benchmark` (~45s). "P(cue visit) by trial" shows the AI/Combined drop.
> - Scenario → `Active Learning — no cue` → `Clear Results` → `Run Benchmark`. "P(risky visit) by trial" shows the AL/Combined decay.
> - Scenario → `Volatile Context` → `Clear Results` → `Run Benchmark`. Flat cue usage — context resamples so the cue is always informative.

**15 · What I'd push back on** (75s)
- **The A-matrix parameterisation assumes known structure.** The agent only learns the *entries* of one column — it already knows the column exists and what its support is. Real perception (drone sensor, camera feed) has to learn the column itself from scratch.
- **Novelty is count-based, not estimate-based.** After a handful of visits the KL drops sharply regardless of whether E[θ] is close to the truth. Curiosity can turn off *before* you've actually learned the right value.
- **Policy space scales combinatorially.** τ = 2 with 2 actions is trivial; τ = 5 in a grid world is 25⁵ and you have to evaluate *G* for every policy. Amortisation via a neural policy is where the field has gone since 2019, and where I'd want to push next.
- None of these are fatal — they're the hinges where classical active inference meets modern deep-learning machinery.

### Part 4 · Extensions + bridge (slides 16–22, ~8 min)

**16 · Grid-maze** (60s)
- 10×10 grid with fog-of-war observations. Now the A-matrix is a *learned map* of the environment rather than a fixed one-column likelihood.
- Salience actually matters here — visiting a cell reduces state-entropy over its contents. Novelty still fires for cells the agent hasn't learned yet. Both EFE terms are now doing real work, not just one of them.
- τ = 2 still runs fine; τ = 5 is where the combinatorics start to bite and amortisation becomes a real need.

**17 · Unknown site search (v2)** (60s)
- 15×15 randomised site — buildings occlude line-of-sight, mix of real targets and decoys, and the agent has **no prior map**.
- Belief state is richer: 4-class belief per cell (unknown / empty / building / target), plus a per-object target-vs-decoy belief.
- Cross-episode **transferable Dirichlet prior α** — concentration over class priors is learnt *across* episodes, not reset each time. Same digamma machinery, but the hypothesis space is now rich enough to do real exploration over layout statistics.

> **Demo cues.** Scenario → `Unknown Site Search` (~2s build).
> - `Run 10` → drone surveys; heatmap saturates; ribbon cycles Explore/Scan/Confirm.
> - `Reset` (new layout, α persists) → `Train 10 eps` → α bars accumulate; mini curve fills.
> - `Reset α` to wipe if re-running cold.

**18 · Ablation** (75s)
- The ablation: zero out *w_salience* and *w_novelty*, clamp β = 0.125. 30 episodes per agent. Isolates the contribution of the curiosity terms without changing anything else.
- **Combined drops from 83 → 47 (−37pp). AI drops 80 → 47 (−33pp).** Every "smart" agent collapses to the same extrinsic-only baseline, because stripping curiosity *is* what removes the advantage. That gap is the empirical proof that the EFE terms are doing the work.
- **The counter-example is important**: Active Learning goes from 27 → 47 — an *anti-gap* of +20pp. Novelty without salience sends the agent roaming blindly because it has no state-info-gain to anchor what it's exploring *for*. Forcing the greedy commit-to-best path rescues it by shutting the roaming down.
- Controls behave as expected: greedy stays around 40, random around 30, both flat under the toggle.

> **Demo cues.** Mode → `Experimenter`.
> - Episodes = 10, ablation **off** → `Run Benchmark` (~60 s). Combined ~83, AI ~80.
> - `Clear Results` → flip ablation **on** → `Run Benchmark` → smart agents collapse to ~47%; controls unchanged.

**19 · What each gap tells us** (60s)
- Combined −37, AI −33: **salience is doing most of the per-episode lifting** — within-episode resolution of state ambiguity. Novelty's role is more about cross-episode transfer of parameter knowledge.
- The AL anti-gap tells us curiosity has to be **aimed**. Novelty alone asks *"what haven't I seen?"*; salience asks *"what would resolve what I care about?"*. Undirected novelty without an extrinsic anchor just roams. You need both.
- One-liner for the room: full agent roughly **doubles greedy** on this task — 83 vs 40.

**20 · Munin Dynamics** (75s) · Feb–Aug 2025, volunteer
- The problem: detect an incoming quadcopter at 200 m range, 150 m/s closing speed, on a sub-30 mm camera board with a $250 bill-of-materials budget. Interceptor-class hardware.
- What I built: camera and SBC selection against a Johnson-criterion budget; FOMO detector, int8-quantised, running on the Digi CC93 SBC with the full capture → detect → track pipeline on-platform.
- The headline number: **90% recall with FOMO** versus mAP50 0.27 for YOLOv11n at the same compute budget. ~200 KB model, under 1 ms inference at 128×128.
- The ceiling of that system is fundamentally *a better detector*. What I want to work on next is the *planner* above it — which is exactly where active inference lives.

**21 · Frazer-Nash** (75s) · Sept 2022 – Apr 2024
- Kargu-class multirotor with EO/IR sensors in a maritime context. Tight perception-to-control loop running on SWaP-constrained hardware.
- What I delivered: rebuilt the vision stack from **20 to 70+ fps** with better accuracy on the same hardware. Led technical design on a **£235k bid** that won.
- Where it broke: under sensor ambiguity, because **confidence is not the same as uncertainty** — a CNN's softmax tells you nothing about whether it knows what it's looking at. The missing piece was a shared uncertainty-aware plane between perception and control, which is precisely what the generative-model framing provides.

**22 · Where it meets Stanhope** (75s)
- I've shipped perception-to-control twice now: pixels to target in milliseconds, but no posterior, domain-specific retraining, greedy downstream policy. The advantages and the limits of that pattern are well known to me.
- Active inference adds what both systems lacked: an explicit **q(θ)** over the world; salience and novelty planned *ahead* rather than post-hoc; one single objective covering both pursuit and exploration; and crucially, self-terminating curiosity that doesn't need a decay schedule.
- Stanhope's Real World Model sits exactly at this layer. And Frazer-Nash SENTINEL — RL for sensor management — is the same problem shape I've been working on.
- **The offering is straightforward**: I've built the perception pipeline *underneath* twice over. I'm here for the agent on top.

**23 · Thanks** (15s)
- The live sim and all source code are linked on the slide. Happy to take questions.

---

### Runtime & trim

| Section | Slides | min |
|---|---|---|
| Open + roadmap | 1–2 | 1.0 |
| Model | 3–6 | 4.5 |
| Planning | 7–11 | 6.5 |
| Caveats + sim | 12–13 | 3.0 |
| Results | 14–15 | 2.5 |
| Extensions + ablation | 16–19 | 4.25 |
| Stanhope bridge | 20–22 | 3.75 |
| Close | 23 | 0.25 |
| **Total** | | **~25.75** |


### Live-demo pre-flight

- **Open `localhost:5173` ≥5 min before** — splash must say "Ready." and fade.
- **Throwaway `Step`** in each scenario — first step JITs the Python path.
- **Experimenter:** Episodes = 10, Batch = 5 for live runs. Full 30-ep benchmark takes 2–3 min — don't run that live.
- **Gotchas**:
  - `Reset` in Experimenter wipes the leaderboard (not a soft reset) — use `Clear Results` between ablation off/on.
  - `Step` relabels to `Run Batch` in Experimenter — it runs a batch, not one step.
  - Don't resize the window mid-demo (training-curve canvas doesn't reflow).
  - If Pyodide wedges after long idle, reload — 10 s recovery.
