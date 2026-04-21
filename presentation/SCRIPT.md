# Speaker script · Stanhope interview presentation

**Target runtime:** ~25 minutes (within the 20–30 min band).
**Pace:** ~145 wpm conversational, ~3,600 words total.
**Philosophy:** speak to the insight, let the slide carry the detail. Do not read bullets.

---

## 1 · Title (≈ 30 s)

Thanks for having me. This is Schwartenbeck et al., 2019 — "Computational mechanisms of curiosity and goal-directed exploration." It's a short paper with a big claim: that curiosity doesn't need to be added to a reinforcement-learning agent as a bonus; it falls out of a single variational objective, for free. I want to walk you through why that claim is true, where it bends, and what it has to do with the drone work I've been doing.

## 2 · Roadmap (≈ 30 s)

Four parts. First, the model — the T-maze task and its generative structure. Second, planning — the expected-free-energy decomposition that is the heart of the paper. Third, evidence — I'll show a live reproduction of Figures 3 and 6, and an ablation I ran. Fourth, how this connects to what I've shipped on defence UAS — and why I think it's the layer above the work I've been doing.

## 3 · The T-maze task (≈ 60 s)

The task is deliberately boring. Agent at the centre; left arm gives a small deterministic reward; right arm gives a large reward with some unknown probability *p*. One trial, one choice, *N* trials total.

The reason this is a useful paper — not a real-world one, a *useful* one — is that it isolates the thing we care about. There's no partial observability, no hidden context, no state estimation to worry about. The only uncertainty in the entire world is a single scalar *p* in the observation model. So any exploratory behaviour you see *has* to be explained by the agent's treatment of that parameter. You can't hide exploration in a sampling schedule or an entropy bonus — they're absent.

## 4 · The generative model (≈ 90 s)

Quick calibration, because this is the hinge between active inference and model-free RL.

A generative model is a joint distribution over observations, states, and parameters. You can sample from it — pick an *s* and a *θ* and it will tell you what observation to expect. That's the *generative* part. Contrast that with Q-learning, which just maps observation to decision — it has no opinion on how the data came to be.

The reason generative models matter here is what I've written on the right: *one model, three jobs*. You use it first for inference — invert the joint with Bayes' rule to get a posterior over states. You use it for simulation — roll it forward under a candidate policy to get a predicted observation distribution. And you use it for meta-inference — reason about *θ* itself, which is how you ask *"will this action teach me something?"*

Model-free RL can't do any of those. There's no *p(o | s, θ)* to invert or roll forward. Curiosity has nowhere to live. This asymmetry is the reason the paper works the way it does.

## 5 · Figure 2 · four pieces (≈ 60 s)

The model itself is a discrete POMDP with four pieces. *A* is the observation model — four observations, three states, a 4×3 matrix. *B* is the transition model, one per action. **c** is the log-preference vector — notice large reward has log-preference of +4, no reward is –2. Preferences are just expected outcomes the agent finds surprising *in a good way*. And **d** is the initial state prior, which just says "you start at the centre."

The one thing I want you to notice on this slide: out of A, B, c, d, only one column of *A* is uncertain. Everything else is deterministic and known. So the whole learning problem — the whole paper — is about what happens to that one risky column.

## 6 · The A-matrix · where uncertainty lives (≈ 75 s)

Zoomed in. Two columns of *A* are deterministic: if you're at centre, you see the centre cue; if you're at safe, you see the small reward. One column — risky — has a single unknown *p*.

The agent's belief over that column is a **Dirichlet**. The Dirichlet is the conjugate prior to the categorical — it's parametrised by concentration counts α, and its expected value is the normalised count vector. The paper starts every agent at α equals [1, 1] — flat, maximum a priori uncertainty.

On the right you can see what happens as counts accumulate. Two observations of *large* gives you Dir(3,1), E[*p*] = 0.75. Six-and-two gives you Dir(7,3), E[*p*] = 0.7, but tighter. That sharpening — the shrinking of the variance — is what the novelty term of the EFE picks up on. Hold on to that.

## 7 · Inference · variational free energy (≈ 90 s)

Exact inference over state trajectories in a POMDP is intractable — it's a sum over all state sequences, which scales combinatorially. So the paper uses a variational substitute. Free energy, as written here, is a KL-like bound on the log-evidence; minimising it gives you the best factorised approximation to the posterior.

When you do that minimisation — constrained to the simplex because *q(s)* has to sum to one — what comes out is the boxed update. A softmax of three log-terms: the likelihood of what you just saw, the forward message from the previous belief, and the backward message from the next one.

If you've ever written a forward-backward pass for an HMM, this is exactly that — just dressed in active-inference notation. The softmax isn't chosen because it's nice; it drops out of the Lagrangian for a simplex-constrained optimisation. It's the only thing it *could* be.

The figure on the left makes it concrete: the prior times the likelihood, renormalised, gives you the posterior. Discrete Kalman.

## 8 · Planning · expected free energy (≈ 90 s)

Now the jump. We can't just minimise *F* to plan — future observations don't exist yet. So we take the expectation under the generative model of what we'd see if we pursued policy *π*. That's expected free energy, *G*.

With a mean-field factorisation between states and parameters and a bit of algebra, *G* decomposes into three terms that are each interpretable on their own.

One — extrinsic cost. "How badly does this policy score against my preferences?" That's the log-preference expectation. That's the only term a greedy RL agent sees.

Two — salience. The KL between my belief about the state *after* observing the outcome, and before. It's state-level information gain.

Three — novelty. The KL between my belief about *θ* after updating, and before. It's parameter-level information gain.

Minus signs on all three. Minimising *G* means: low cost, high information gain. Curiosity is not added. It *falls out* of the same inequality that delivers reward-seeking. That's the whole argument of the paper on one slide.

## 9 · The three terms · what each buys you (≈ 75 s)

Each term does something distinct. Extrinsic — top-left panel — is a weighting over log-preferences. Large reward is +4, no reward is –2, centre and small are in the middle. This is the "scoring" term.

Salience — middle panel. If a hypothetical observation would *concentrate* my belief about the state, that's information gain over the state. In the bare T-maze there's nothing to be salient about — the state is observable — so this term is near-zero. It comes alive in the grid and drone extensions I'll show.

Novelty — right panel. This is the one that makes curiosity work here. A Beta density before observing the outcome, and a sharper Beta after. The area between them is what the novelty term captures. And crucially — this is the bit most ML people fudge — this KL has a **closed form** because of conjugacy. No sampling. No amortiser. Digamma functions evaluated at integer counts. I'll come back to that in two slides.

## 10 · Parameter learning · why Dirichlet (≈ 75 s)

The conjugate pair. Categorical likelihood times a Dirichlet prior gives you a Dirichlet posterior with the counts incremented. The update rule is just: see observation *o* in state *s*, add one to α at that index. That's it. That's the learning rule.

Two moments matter. The expected value of θ is the normalised count. And the expected *log* of θ is the difference of two digammas. That second identity is what makes the novelty term computable without sampling, without variational approximation — it's a closed-form function of integer counts.

Why this matters: if you've tried to put information-gain terms into a deep-RL agent, you know that estimating them is painful and noisy. Here it's exact. That's not a convenience; it's what makes the whole EFE calculation tractable for this paper. If you lose conjugacy, the derivation breaks.

## 11 · Policy selection · precision (≈ 60 s)

Given *G* for each policy, the policy posterior is itself a softmax. The temperature is *γ* — called precision — and the paper puts a Gamma prior on *γ* with rate *β*.

*β* is the lever that separates the paper's four experimental regimes. Small *β*, high precision — near-argmin over *G*, the agent commits. Large *β*, low precision — near-uniform, the agent is random.

The reason to put a prior on *γ* at all, rather than fixing it, is to let the agent modulate its own confidence. When candidate policies have close *G* values, *γ* naturally shrinks — the agent hedges. It's saying *"I genuinely don't know, so don't commit."* The Gamma–Dirichlet pairing is what lets all of this stay closed-form.

## 12 · Four things the paper doesn't tell you (≈ 60 s)

A short aside. When you implement this paper, four things aren't on the page. Policy horizon — the paper uses τ equals two, one step. I implemented full τ-lookahead but kept τ = 2 to replicate. Expected observation posterior — you can either sample or do an analytic marginal with a plug-in mean for *A*. I use the analytic version. Numerical safety — log-zero is a real risk with sparse *A*, so I clamp with a small ε, same as the reference `spm_MDP_VB` code. And the novelty term has a subtle thing about whether you take expectation over *q(A)* or just plug in the mean; the difference is small when α₀ is large, but not when it's small. I'll happily go deeper on any of those.

## 13 · Sim · T-maze end-to-end (≈ 120 s — includes live demo)

This is the sim. Python running in the browser via Pyodide, same code I run locally. Everything on the screen you're looking at is the paper's agent.

Let me walk through. The four regimes from the paper are in the top dropdown — active-inference, active-learning, greedy, random. The three weights on the left — extrinsic, salience, novelty — let you turn EFE components on and off. *β* is the policy precision.

**[Click to active_learning.]** Active-learning agent. Notice it pulls the risky arm early — salience is zero in this task so that's pure novelty driving it. And watch the sensor-model panel — the Dirichlet concentrations are updating in real time. As α₀ grows, novelty decays. You can see the agent stop exploring once its belief is sharp enough.

**[Click to random, β=8.]** Random agent for contrast. It never stops exploring — there's no self-termination because exploration isn't tied to what's been learned.

**[Click to greedy.]** Greedy, with a pessimistic prior. It never even discovers the risky arm is good. Curiosity matters because without it, an agent's *initial* belief determines its *final* behaviour.

The URL's at the bottom of the slide if you want to play with it later.

## 14 · The key result · Figure 6 (≈ 60 s)

This is Figure 6 from the paper — the single plot that carries the argument. Three curves. Active-learning agent: probability of choosing risky *decays* as uncertainty resolves — the curiosity term self-terminates. Random agent: flat, because its exploration has nothing to do with what it's learned. Greedy agent: stuck — if its prior was unlucky, it never revisits.

What this plot tells us: exploration collapses *from inside* the same objective that drives exploitation. No separate schedule, no epsilon-greedy, no entropy bonus. And because the decay is governed by the digamma of α₀, it's a closed-form function of time — you can actually compute when curiosity will turn off.

This is the paper's strongest claim, and it replicates. I've rerun this in the sim.

## 15 · What I'd push back on (≈ 75 s)

I want to be honest about three things I'd push back on.

One. The A-matrix parameterisation assumes known structure. The agent learns the *distribution* over outcomes in the risky column, but it already knows the column *exists* — it knows there are two arms, it knows what "risky" means. Real perception on a drone has to learn the structure, not just the entries.

Two. Novelty is count-based, not estimate-based. After visiting a state twice, the KL between posterior-after and posterior-before has already dropped substantially, *regardless of whether the estimate is close to the truth*. Curiosity turns off before you've actually learned. You can construct adversarial environments where this matters.

Three. Policy space scales combinatorially. τ=2 with two actions is fine; τ=5 in a grid world is 25 to the fifth policies, and you're evaluating *G* for each one. Amortising this — learning a policy network that outputs the posterior directly — is the obvious next move, and it's where most of the field has gone since 2019.

None of these are fatal. They're the hinges for where active inference has gone since this paper.

## 16 · Extension 1 · grid-maze (≈ 60 s)

Quickly, two extensions I built to stress-test the decomposition.

First: a 10×10 grid with fog of war. Goal hidden at an unknown cell. The agent's *A* matrix is now a learned map — "what do I expect to see at cell *s*". Salience now *matters*, because visiting a cell reduces state-entropy about its contents. Novelty still fires for cells the agent hasn't seen.

In the sim, the behaviour extends qualitatively — the novelty term decays, coverage increases, path cost is bounded. But you start to feel the combinatorics. τ=2 is fine; τ=5 is a noticeable wait.

## 17 · Extension 2 · drone search (≈ 60 s)

Second extension — closer to where I've been working. 3D grid, obstacles, a drone with a camera frustum, so partial field of view. Belief per cell over target-present versus target-absent. The policy is now two-dimensional: where to fly, and where to look.

The salience term, in this world, picks viewpoints that maximally reduce uncertainty over target locations. That is essentially **next-best-view planning** — but derived from a single free-energy objective, not hand-coded.

I want to be honest: this is didactic. A real system needs amortised policy posteriors — a neural network that maps belief to a policy distribution — because you cannot enumerate *G* over continuous motion. But the *structure* of the objective carries over.

## 18 · Ablation · isolating the curiosity contribution (≈ 75 s)

An experiment I ran to answer: if you strip curiosity out — zero salience, zero novelty, greedy precision — how much worse do you get?

You see four bars per row: "with curiosity" in green, "ablated" in red. Combined agent: 88 down to 68 — a 20-point drop. Active-inference, salience-only: 92 down to 66 — 26 points. Active-learning, novelty-only: 89 down to 57 — 32 points.

Controls at the bottom: greedy and random already had their curiosity weights off and their β fixed. The toggle is a no-op for them, and indeed their scores barely move. That's the control — the switch isolates curiosity rather than degrading policy.

The headline: curiosity earns the last 20 to 32 percentage points. And the gap is largest for active-learning, because active-learning is the agent most dependent on iteratively building sensor knowledge.

## 19 · What each gap tells us (≈ 60 s)

Short gloss on each gap. Combined minus 20: full EFE surveys, scans top candidates twice at altitude two, confirms at altitude one. Ablated, only signal is confirm extrinsic — agent wanders until an accidental sighting pushes it over threshold; sometimes commits to a distractor.

Active-inference minus 26: salience drives the scan-then-confirm tactic. Strip it, there's no mechanism for "go look closer at object X". Single-component designs break hardest under ablation.

Active-learning minus 32: novelty validates *which altitude discriminates*. Without novelty, the agent never figures out that altitude two is the discriminating altitude. It's the biggest loser in small-batch runs because it depends on iteratively-built sensor knowledge that never materialises.

The one-liner — curiosity lets a drone *seek* information, not just exploit what passive observation hands it. Strip it, you lose 20 to 32 points.

## 20 · Drone work at Munin Dynamics (≈ 75 s)

Quickly — what I've been doing recently. Munin Dynamics — I joined as a volunteer engineer in February. The bet is the cheapest, smallest missile system that can down an FPV drone. Pocket-sized. Scalable to Ukraine-level threat volumes.

The perception constraints are fierce. Detect a quadcopter out to 200 metres; closing velocity 150 m/s; a 30 mm diameter processing board; $250 production BoM.

What I built: camera and SBC selection against Johnson's criteria; a FOMO detector trained on-platform, int8-quantised; full capture-detect-track pipeline on a Digi ConnectCore 93.

The numbers: FOMO hits 90% recall where YOLOv11n delivered 0.27 mAP50 at the same compute. Model size is 200 kilobytes versus five megs. Inference under a millisecond at 128 by 128. BoM comfortably inside the budget.

This is a beautiful applied problem. And the reason I'm sitting here and not there is that the *ceiling* of that system is a better detector. The work I want to be doing is the planner above it.

## 21 · Before Munin · autonomous UAS at Frazer-Nash (≈ 75 s)

Before Munin, I spent a year and a half at Frazer-Nash on a tactical multirotor — the nearest public analogue is STM's Kargu. Mini-class, EO/IR, loitering ISR with precision strike, adapted for maritime defence. Onboard vision, tracking, navigation on SWaP-constrained hardware, in real time.

Architecturally it was a tightly coupled perception–control loop. Detector plus tracker produced target estimates; control acted on them immediately. It worked well when observations were clean. I rebuilt the vision stack from 20 to 70-plus frames per second on edge hardware while improving detection accuracy, and I led the technical design on a new £235k bid.

Where it broke was under ambiguity. Clutter, sun-glint, partial occlusion. The detector was *confident* when it shouldn't have been — confidence was not the same as uncertainty. The system had no explicit belief over the environment, no way to reason about what it didn't know, no mechanism to trade off exploration versus exploitation.

The core gap wasn't perception or control individually. It was the absence of a shared, uncertainty-aware layer linking them. That missing layer is what drew me to active inference, and to this paper.

## 22 · Where this work meets Stanhope (≈ 75 s)

Two perception stacks, same ceiling. Both were discriminative layers. Both broke when confidence stopped meaning *what the system actually knows*.

On the left — what I've shipped, twice. Pixels to target, in milliseconds, on SWaP-tight hardware. But no posterior. No notion of "I don't know". Retrain per domain. Greedy downstream consumer.

On the right — what active inference adds. An explicit *q(θ)*, a budget for what's unknown. Salience and novelty planned *ahead*, answering "where should the sensor look next?". One objective driving pursuit *and* exploration — no epsilon-greedy hack. Self-terminating curiosity, in closed form.

Stanhope's Real World Model is pitched at exactly this layer — adaptive, uncertainty-aware behaviour on small autonomous platforms, trialled in drone and robotics partnerships. Frazer-Nash's SENTINEL — an RL-for-sensor-management system I'd be happy to talk about separately — is the same problem shape; active inference derives it from one variational objective instead of a hand-engineered reward.

The offering is simple. I've shipped the perception pipeline that sits *underneath* a Stanhope-style agent on a defence UAS — twice. What I came here for is the agent on top.

## 23 · Thanks (≈ 15 s)

Sim and source are on the slide. I'll stop there — happy to take questions.

---

## Runtime check

| Section | Slides | Minutes |
|---|---|---|
| Opening + roadmap | 1–2 | 1.0 |
| The model | 3–6 | 4.5 |
| Inference + planning | 7–11 | 6.5 |
| Caveats + sim | 12–13 | 3.0 |
| Results | 14–15 | 2.25 |
| Extensions + ablation | 16–19 | 4.25 |
| Bridge to Stanhope | 20–22 | 3.75 |
| Close | 23 | 0.25 |
| **Total** | | **~25.5 min** |

## Cuts if short on time

- Slide 12 (four things paper doesn't tell you) — skip entirely, fold into Q&A (-1 min)
- Slide 16 (grid-maze extension) — collapse into one sentence inside slide 17 (-45 s)
- Slide 19 (what each gap tells us) — compress to 20 s headline (-40 s)
- Drop ablation detail on slide 18 — keep only headline number (-30 s)

That's a clean ~22 minute version if you get signalled to wrap.

## Expansions if you have time

- Dwell on the Dirichlet conjugacy (slide 10) — the digamma identity is a natural place to be asked questions
- Extra pass on the sim (slide 13) — walk through the sensor-model panel updating live
- On the bridge slide (22) — land harder on SENTINEL as a direct point of contact
