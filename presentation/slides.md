---
theme: default
title: Computational mechanisms of curiosity and goal-directed exploration
info: |
  Presentation on Schwartenbeck et al. (2019) for Stanhope AI.
  Jake Watson — April 2026.
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
fonts:
  sans: Inter
  serif: 'Source Serif Pro'
  mono: 'JetBrains Mono'
---

# Computational mechanisms of curiosity<br/>and goal-directed exploration

Schwartenbeck, Passecker, Hauser, FitzGerald, Kronbichler, Friston · *eLife* 2019

<div class="pt-12 opacity-70">
Jake Watson · Stanhope AI · April 2026
</div>

<!--
Plan for the next ~25 min:
- Setup the task and the generative model (Figure 2)
- Derive the variational inference updates
- Derive expected free energy and its three exploratory components
- Parameter learning via Dirichlet conjugacy
- Live reproduction in the browser
- One honest criticism
- Brief extensions + drone work
-->

---

# Roadmap

<div class="grid grid-cols-2 gap-8 pt-4">

<div>

**Part 1 · The model**
- T-maze task and generative model *(Fig 2)*
- State inference: variational free energy
- Parameter learning: Dirichlet updates

**Part 2 · Planning**
- Expected free energy decomposition
- Extrinsic · salience · novelty
- Policy posterior with precision $\gamma$

</div>
<div>

**Part 3 · Evidence**
- Live reproduction of Fig 3 and Fig 6
- Where I think it breaks

**Part 4 · Bridging to Stanhope**
- Grid-maze + drone search extensions
- Micro-MANPAD perception work

</div>
</div>

---

# The T-maze task

<div class="flex justify-center pt-2">

```mermaid {scale: 0.7}
graph TD
  S[Start] -->|a1| Safe
  S -->|a2| Risky
  Safe[Safe arm<br/>+small reward<br/>always]
  Risky[Risky arm<br/>+large w.p. p<br/>-no-reward w.p. 1-p]
```

</div>

<div class="grid grid-cols-2 gap-10 pt-4 max-w-5xl mx-auto text-sm">

<div>

<v-clicks>

- Agent starts at centre
- Left arm: **safe** — small reward, deterministic
- Right arm: **risky** — large reward with unknown probability $p$
- One trial = one arm choice
- $N$ trials to maximise total reward

</v-clicks>

</div>

<div class="opacity-80">

<div v-click>

The agent does not know $p$. It must choose between exploiting what it knows (safe) and learning what it doesn't (risky) — without an explicit exploration bonus.

</div>

<div class="pt-3" v-click>

**Why this task?** It isolates the learning problem. No hidden context, no partial observability — the only uncertainty is about $p$, a parameter of the observation model.

</div>

</div>

</div>

---

# The generative model

<div class="grid grid-cols-2 gap-8 pt-2">

<div>

A **generative** model is a joint distribution that says *how the world produces observations*:

$$
p(o, s, \theta) \;=\; \underbrace{p(o \mid s, \theta)}_{\text{likelihood}}\;\underbrace{p(s)}_{\text{state prior}}\;\underbrace{p(\theta)}_{\text{parameter prior}}
$$

Read it as: *"if the world were in state $s$ with sensor parameters $\theta$, this is the distribution over observations I'd expect to see."* It is generative because you can **sample from it** — pick an $s$ and $\theta$, the model tells you what $o$ would result.

<div class="pt-3 text-sm opacity-80">

Contrast with a **discriminative** model like Q-learning or a classifier, which maps observation → decision without representing how data came to be.

</div>

</div>
<div>

**Why it matters: one model, three jobs.**

<div class="pt-1 text-sm">

1. <span class="text-sky-400 font-bold">Inference.</span> Invert the joint: $p(s \mid o) \propto p(o \mid s)\,p(s)$. Salience needs this.

2. <span class="text-emerald-400 font-bold">Simulation.</span> Roll forward under a policy to get $q(o \mid \pi)$. Extrinsic scores it against preferences.

3. <span class="text-violet-400 font-bold">Meta-inference.</span> Reason about $\theta$ itself — *how much would observing here sharpen my belief about the sensor?* Novelty is a KL on $q(\theta)$.

</div>

<div class="pt-3 text-sm opacity-80">

Model-free RL has none of these — there is no $p(o \mid s, \theta)$ to invert or roll forward, so *there is nowhere for curiosity to live*. This is the asymmetry that defines active inference.

</div>

</div>
</div>

<!--
The prose bridge between the T-maze setup and the tensor slide. Key line to drop if asked: "same generative model is used three times — to predict what will be observed, to score information gain, and to evaluate preferred outcomes." One model, three jobs.
-->

---

# Figure 2 · four pieces of the model

A discrete POMDP with four pieces: $\{A, B, \mathbf{c}, \mathbf{d}\}$

<div class="grid grid-cols-2 gap-8 pt-2 text-sm">

<div>

**States** $s \in \{\text{start, safe, risky}\}$  (3)

**Observations** $o \in \{\text{centre, small, large, none}\}$  (4)

**Actions** $a \in \{\text{go-safe, go-risky}\}$  (2)

$$
A_{o,s} = P(o \mid s)
\quad\text{(4×3, observation model)}
$$

$$
B^{(a)}_{s',s} = P(s' \mid s, a)
\quad\text{(3×3×2, transitions)}
$$

</div>
<div>

**Preferences** $\mathbf{c} \in \mathbb{R}^{|O|}$ *(log-space)*
$$
\mathbf{c} = [\,0,\ 2,\ 4,\ -2\,]^\top
$$
so $P(o \mid C) \propto \exp(\mathbf{c})$.

**Initial state prior**
$$
\mathbf{d} = [\,1,\ 0,\ 0\,]^\top
$$

<div class="pt-2 opacity-70">
The risky column of $A$ is the only uncertain object in the model. Everything else is deterministic and known.
</div>

</div>
</div>

---

# The A-matrix · where uncertainty lives

<div class="grid grid-cols-2 gap-8 pt-1">

<div>

$$
A \;=\;
\begin{array}{c|ccc}
 & s{=}\text{start} & s{=}\text{safe} & s{=}\text{risky} \\ \hline
o{=}\text{centre} & 1 & 0 & 0 \\
o{=}\text{small}  & 0 & 1 & 0 \\
o{=}\text{large}  & 0 & 0 & p \\
o{=}\text{none}   & 0 & 0 & 1-p
\end{array}
$$

<div class="pt-3 text-sm">

Two deterministic columns (start, safe). One **unknown** column (risky), parametrised by a single scalar $p$.

Belief over the risky column is a **Dirichlet** with concentrations $\boldsymbol{\alpha} \in \mathbb{R}^{2}_+$:

$$
q(A_{\cdot,\text{risky}}) = \text{Dir}(\boldsymbol{\alpha}), \quad
\mathbb{E}[A_{o,\text{risky}}] = \frac{\alpha_o}{\alpha_0}
$$

Paper starts at $\boldsymbol{\alpha}{=}[1,1]$ — maximum a priori uncertainty.

</div>

</div>
<div class="bg-white rounded-lg p-2 self-center">

<img src="/beta-sharpening-basic.png" class="w-full" />

</div>

</div>

---

# Inference · variational free energy

Exact Bayesian inference $q(s \mid o)$ is intractable in POMDPs. Variational substitute:

$$
\mathcal{F}(\pi) \;=\; \mathbb{E}_{q(s\mid\pi)}\!\left[\ln q(s\mid\pi) - \ln p(\tilde o, s \mid \pi)\right]
$$

Minimising $\mathcal{F}$ yields the **message-passing update** — for each time-step $\tau$:

$$
\boxed{\ q^*(s_\tau \mid \pi)
= \sigma\!\Big(
\underbrace{\ln A^{\!\top}\tilde o_\tau}_{\text{likelihood}}
\;+\; \underbrace{\ln B_{\tau-1}(\pi)\, q^*(s_{\tau-1} \mid \pi)}_{\text{forward}}
\;+\; \underbrace{\ln B_\tau(\pi)^{\!\top}\, q^*(s_{\tau+1} \mid \pi)}_{\text{backward}}
\Big)
\ }
$$

<div class="grid grid-cols-5 gap-6 pt-3">

<div class="col-span-3 bg-white rounded-lg p-2">

<img src="/posterior-update.png" class="w-full" />

</div>
<div class="col-span-2 text-sm">

- $\sigma(\cdot)$ = softmax — falls out of constrained minimisation over the simplex (Lagrange + $\sum q = 1$)
- $\ln A^{\!\top} \tilde o$ is **log-evidence** arriving at state $s$
- This is forward-backward in active-inference clothing — a single observation multiplies the prior pointwise and renormalises, exactly like the Kalman update in discrete form

</div>
</div>

<!--
If asked: where does softmax come from? Answer: minimise F subject to Σq = 1 (simplex). Lagrangian → exponential form → normalisation → softmax. Not an arbitrary choice.
-->

---

# Planning · expected free energy

For actions we can't just minimise $\mathcal{F}$ — future observations aren't observed yet. We take an expectation:

$$
G(\pi, \tau) \;=\; \mathbb{E}_{q(o_\tau,\, s_\tau,\, \theta \mid \pi)}\!\left[
\ln q(s_\tau, \theta \mid \pi) \;-\; \ln p(o_\tau, s_\tau, \theta \mid \pi, C)
\right]
$$

where $\theta = A$ (the unknown parameters) and $C$ encodes preferences via $\mathbf{c}$.

<div class="pt-2">

Under a mean-field factorisation $q(s,\theta \mid \pi) = q(s \mid \pi)\, q(\theta)$ and expanding the joint,
we can decompose this into **three interpretable terms**:

</div>

$$
G(\pi, \tau) \;=\;
\underbrace{-\,\mathbb{E}_{q(o_\tau\mid\pi)}\!\left[\ln p(o_\tau \mid C)\right]}_{\text{extrinsic / pragmatic cost}}
\;-\; \underbrace{\mathbb{E}_{q}\!\left[D_{\text{KL}}\!\big[q(s_\tau\mid o_\tau,\pi)\,\|\,q(s_\tau\mid\pi)\big]\right]}_{\text{salience (state info gain)}}
\;-\; \underbrace{\mathbb{E}_{q}\!\left[D_{\text{KL}}\!\big[q(\theta\mid s_\tau, o_\tau)\,\|\,q(\theta)\big]\right]}_{\text{novelty (parameter info gain)}}
$$

<div class="pt-4 text-sm opacity-80">

The minus signs matter: minimising $G$ means **low expected cost** *and* **high expected information gain**. Exploration is not a bonus grafted on — it drops out of the same objective as reward-seeking.

</div>

---

# The three terms · what each one <em>buys</em> you

<div class="bg-white rounded-lg p-2">

<img src="/three-terms-panel.png" class="w-full" style="max-height: 42vh; object-fit: contain;" />

</div>

<div class="grid grid-cols-3 gap-5 pt-1 text-xs">

<div>

**1 · Extrinsic** — exploitation. Scores expected outcomes against preferences $\mathbf{c}$. The only term a greedy RL agent optimises.

</div>

<div>

**2 · Salience** — state-info gain. KL between belief *after* seeing $o$ and *before*. Vanishes in the bare T-maze (no cue).

</div>

<div>

**3 · Novelty** — parameter-info gain. KL on $q(\theta)$. Drives sampling where the sensor is uncertain. **Self-terminating** as $\boldsymbol{\alpha}$ concentrates.

</div>

</div>

---

# Parameter learning · why Dirichlet

<div class="grid grid-cols-2 gap-6 pt-2">

<div class="text-sm">

Posterior over the risky column is categorical-likelihood × Dirichlet-prior:

$$
p(\boldsymbol\theta \mid \mathcal{D}) \;\propto\; \text{Cat}(\mathcal{D}\mid\boldsymbol\theta)\,\text{Dir}(\boldsymbol\theta \mid \boldsymbol\alpha)
\;=\; \text{Dir}\!\left(\boldsymbol\alpha + \mathbf{n}\right)
$$

— **conjugate**. On observing $o$ in state $s$, increment $\alpha_{o,s}$ by one.

<div class="pt-3">

Two moments we need:

$$
\mathbb{E}[\theta_i] = \frac{\alpha_i}{\alpha_0}, \qquad
\mathbb{E}[\ln \theta_i] = \psi(\alpha_i) - \psi(\alpha_0)
$$

with $\psi$ the digamma function and $\alpha_0 = \sum_i \alpha_i$.

The second identity is what makes **novelty** computable in closed form — no sampling, no amortiser. You evaluate digammas at integer counts.

</div>

</div>
<div class="bg-white rounded-lg p-2 self-center">

<img src="/dirichlet-digamma.png" class="w-full" />

</div>

</div>

<!--
This is the bit most ML people fudge. Novelty reduces to a closed-form digamma expression because of conjugacy. Conjugacy is not a convenience, it's what makes the whole EFE calculation tractable in this setting.
-->

---

# Policy selection · precision

Policies are scored by $-G(\pi)$; the policy posterior is itself a softmax:

$$
q(\pi) \;=\; \sigma\!\left(-\gamma \cdot G(\pi)\right), \qquad \gamma \sim \Gamma(1, \beta)
$$

$\gamma$ is **precision** over policies — expected under a Gamma prior with rate $\beta$. $\beta$ is the lever that separates the paper's four experimental regimes:

<div class="bg-white rounded-lg p-2 pt-1 mt-2">

<img src="/softmax-temperature.png" class="w-full" />

</div>

<div class="pt-3 text-sm opacity-85">

**Why a prior on $\gamma$ at all?** It lets the agent *modulate its own confidence*. When $G(\pi)$ values are close, $\gamma$ naturally shrinks — *"I genuinely don't know which is better, so don't commit."* The Gamma–Dirichlet pairing is what lets the whole update stay closed-form.

</div>

---

# Four things the paper doesn't tell you

<div class="text-sm">

A few places you can't read off the paper:

<v-clicks>

- **Policy horizon.** The paper uses $\tau = 2$ (one step). I implemented full $\tau$-lookahead as a sum $\sum_\tau G(\pi, \tau)$ but kept $\tau = 2$ for replication.
- **Expected posterior $q(o \mid \pi)$.** Analytic marginalisation: $q(o \mid \pi) = \sum_s A \cdot q(s \mid \pi)$. Avoids sampling noise at the cost of assuming $A \approx \bar A$ (the plug-in mean).
- **Numerical safety.** $\ln 0$ is a real risk with sparse $A$. I clamp to a small $\varepsilon$, same as the reference `spm_MDP_VB` code.
- **Novelty with $A$-mean vs $A$-sampled.** The analytic form uses $\bar A$; a more honest version takes an expectation over $q(A)$. Difference is small for the paper's configs but noticeable when $\alpha_0$ is small.

</v-clicks>

</div>

<!--
These four choices signal you engaged with the paper as an implementer, not a reader. Surface 1-2 of these unprompted; expect them to ask about the others.
-->

---

# Sim · T-maze end-to-end

<div class="grid grid-cols-5 gap-6 pt-2">

<div class="col-span-3">

<img src="/sim-tmaze.png" class="rounded-lg shadow-xl" style="border: 1px solid rgba(255,255,255,0.1);" />

</div>
<div class="col-span-2 text-sm">

Python (NumPy) model running in the browser via **Pyodide**. Same `.py` files run locally for validation.

<div class="pt-4">

**What maps to what on screen**

- **β (precision)** → policy softmax temperature
- **Extrinsic · Salience · Novelty** → the three EFE weights $w_{\text{ext}}, w_{\text{sal}}, w_{\text{nov}}$
- **Sensor model** panel → the Dirichlet concentrations $\boldsymbol{\alpha}$, updating live
- Agent dropdown toggles the four regimes from the paper
- Scenario dropdown switches between the paper's three T-maze tasks: volatile context, stable context, and no-cue active learning

</div>

<div class="pt-4 text-center">

<a href="https://jakeywatson.github.io/stanhope/sim/" target="_blank" class="inline-block px-5 py-2.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-semibold text-base no-underline shadow-lg">
Open the sim →
</a>

<div class="pt-2 text-xs opacity-50">jakeywatson.github.io/stanhope/sim</div>

</div>

</div>

</div>

<!--
Demo points to hit live:
1. Active-learning agent: show early risky-arm sampling, novelty term high
2. Watch concentration parameters grow, novelty decay
3. Switch to random (β=2³): exploration never decays
4. Switch to greedy: never discovers risky arm even when it's good
5. Resist live-tweaking — stick to the script
-->

<!--
Demo points to hit live:
1. Active-learning agent: show early risky-arm sampling, novelty term high
2. Watch concentration parameters grow, novelty decay
3. Switch to random (β=2³): exploration never decays
4. Switch to greedy: never discovers risky arm even when it's good
5. Resist live-tweaking — stick to the script
-->

---

# The key results · Figure 6 + Figure 8

<div class="grid grid-cols-2 gap-8">

<div>

**Fig 6C — Active Learning (no cue)**

<v-clicks>

- AL & Combined: **P(risky) decays** as Dirichlet over $p$ concentrates
- Random: flat — exploration not tied to learning
- Greedy: stuck if initial $\mathbb{E}[p]$ low — risky column never revisited
- Decay rate is closed-form in $\psi(\alpha_0)$

</v-clicks>

**Fig 8C — Active Inference (stable context)**

<v-clicks>

- AI & Combined: **P(cue visit)** collapses ~1.0 → ~0.1 over 32 trials
- d-vector concentrates → salience info gain → 0 → cue-then-best loses to direct-arm
- Volatile context (Fig 8A): cue usage stays flat — context resamples, cue always pays

</v-clicks>

</div>
<div>

**What this tells us**

<v-clicks>

- Two separate exploratory behaviours — **parameter learning** (Fig 6) and **state-inference cueing** (Fig 8) — both self-terminate from the same $G$ objective
- No exploration schedule, no $\varepsilon$-greedy, no entropy bonus
- You can *compute* when each term switches off: novelty dies with $\psi(\alpha_0)$; salience dies with the d-vector entropy

</v-clicks>

</div>

</div>

<div v-click class="pt-6 text-center text-lg opacity-90">
The sim has three T-maze variants — one per task in the paper — and reproduces both shapes.
</div>

---

# What I'd push back on

<div class="text-base">

<v-clicks>

- **The A-matrix parameterisation assumes known structure.** The agent only learns the *distribution* over outcomes in the risky column — it already knows the column *exists*. Real-world perception (e.g. a drone) has to learn the structure too.

- **Novelty is count-based, not estimate-based.** After visiting a state twice, $D_\text{KL}[\,q(\theta \mid \text{new}) \,\|\, q(\theta)\,]$ has dropped by a large amount, regardless of whether $\mathbb{E}[\theta]$ is close to the truth. Curiosity turns off before you've *actually* learned.

- **Policy space scales combinatorially.** $\tau = 2$ with 2 actions is fine. $\tau = 5$ in a grid world is $25^5$ policies — you evaluate $G(\pi)$ for every one. Amortisation via neural policies is the obvious next move.

</v-clicks>

</div>

<div v-click class="pt-6 text-sm opacity-75">
None of these are fatal — they're the hinges for where the field has gone since 2019 (active inference with neural amortisers, hierarchical generative models, learned structure).
</div>

---

# Extension 1 · grid-maze

<div class="grid grid-cols-2 gap-8 pt-4">

<div>

- 10×10 grid, fog-of-war observation model
- Goal hidden at unknown cell
- Agent's $A$ is a learned map: $P(\text{wall}, \text{open}, \text{goal} \mid s)$
- Salience now *matters* — visiting a cell reduces state-entropy about its contents
- Novelty still fires for uncertain cells

<div class="pt-4 text-sm opacity-80">

Shows the same mechanism extends to **real spatial POMDPs**, not just the 3-state toy.

</div>

</div>
<div>

**Why this matters for Stanhope**

Spatial navigation under partial observability is the default, not the exception. The T-maze version of the paper is didactic — the production question is: does the decomposition still give you exploration-that-terminates when the state space is non-trivial?

In the sim: yes, qualitatively. The novelty term decays, coverage increases, path cost is bounded. But — see prior slide — the combinatorics start biting at $\tau > 2$.

</div>
</div>

---

# Extension 2 · unknown site search

<div class="grid grid-cols-2 gap-8 pt-4">

<div>

- 15×15 site, randomised each episode — **no map**
- Buildings occlude line-of-sight (height-field raycasts)
- 4-class cell belief: **empty · building · decoy · target**
- Per-object target-vs-decoy belief
- **Transferable Dirichlet prior** α — learnt *across* episodes

**Policy**: where to fly · which altitude · when to commit

<div class="pt-4 text-sm opacity-80">

Salience picks viewpoints that reduce cell-class entropy — next-best-view. Novelty rewards cells whose class-prior α is still flat. Hit Train in the sim to watch α accumulate across silent episodes.

</div>

</div>
<div>

**Why this matters**

The v2 world is an *unknown* environment, not a fixed POMDP. Every episode the layout resets but the agent retains Dirichlet counts over *what cell-types look like at this site*. That's the Stanhope-flavoured transferable prior in miniature — same variational objective, generative-model parameters persist.

Caveat still honest: discrete cells, tabular belief. A real system wants neural amortisation. But the structure — three-term EFE, digamma closed form — is unchanged.

</div>
</div>

---

# Ablation · isolating the curiosity contribution

<div class="text-sm opacity-80 pb-2">

Toggle strips curiosity: $w_{\text{sal}} = w_{\text{nov}} = 0$ and collapses $\beta \to 0.125$ (greedy-precise). "If you only chased the extrinsic signal, how well would you do?" — 30 episodes per agent, 400-step cap, v2 unknown-site.

</div>

<div class="pt-1 space-y-2.5 text-sm">

<div class="flex items-center gap-3">
  <div class="w-40">combined</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-emerald-500 rounded" style="width:83%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">83%</div>
  </div>
  <div class="w-6 text-center opacity-50">→</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-rose-500 rounded" style="width:47%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">47%</div>
  </div>
  <div class="w-20 text-rose-300 font-mono">−37pp</div>
</div>

<div class="flex items-center gap-3">
  <div class="w-40">active_inference</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-emerald-500 rounded" style="width:80%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">80%</div>
  </div>
  <div class="w-6 text-center opacity-50">→</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-rose-500 rounded" style="width:47%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">47%</div>
  </div>
  <div class="w-20 text-rose-300 font-mono">−33pp</div>
</div>

<div class="flex items-center gap-3">
  <div class="w-40">active_learning</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-amber-600 rounded" style="width:27%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">27%</div>
  </div>
  <div class="w-6 text-center opacity-50">→</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-amber-400 rounded" style="width:47%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">47%</div>
  </div>
  <div class="w-20 text-amber-300 font-mono">+20pp ⚠</div>
</div>

<div class="flex items-center gap-3 opacity-70">
  <div class="w-40">greedy <span class="text-xs opacity-60">(control)</span></div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-slate-400 rounded" style="width:40%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">40%</div>
  </div>
  <div class="w-6 text-center opacity-50">≈</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-slate-400 rounded" style="width:47%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">47%</div>
  </div>
  <div class="w-20 opacity-60 font-mono">≈ 0</div>
</div>

<div class="flex items-center gap-3 opacity-70">
  <div class="w-40">random <span class="text-xs opacity-60">(control)</span></div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-slate-400 rounded" style="width:30%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">30%</div>
  </div>
  <div class="w-6 text-center opacity-50">≈</div>
  <div class="flex-1 h-5 bg-white/5 rounded relative">
    <div class="absolute inset-y-0 left-0 bg-slate-400 rounded" style="width:17%"></div>
    <div class="absolute inset-0 flex items-center justify-end pr-2 text-xs font-mono">17%</div>
  </div>
  <div class="w-20 opacity-60 font-mono">≈ 0</div>
</div>

</div>

<div class="pt-4 grid grid-cols-12 gap-4 text-xs opacity-75">
<div class="col-span-4"><b>Full EFE</b> · paper-matched weights</div>
<div class="col-span-4 text-center"><b>Ablated</b> · extrinsic only, greedy β</div>
<div class="col-span-4 text-right">15×15 unknown site · 30 eps/agent</div>
</div>

<div class="pt-3 text-sm opacity-85">
Every "smart" agent ablates to the same ~47% extrinsic-only baseline — which <i>is</i> the point: <b>curiosity is what distinguishes them</b>. Salience (AI) and salience+novelty (combined) buy 33–37pp above it. Active-learning (novelty-only) sits <i>below</i> — the counter-example that makes the lesson sharp: diffuse information-seeking, without salience's targeting, is worse than no seeking.
</div>

<!--
Headline: combined +37, AI +33 above a shared ablated baseline (~47%). AL inverts — novelty alone is worse than pure greedy exploitation.
If asked why AL is worse: novelty rewards unvisited cells regardless of whether they matter for the task. Salience is what aims the information-gathering. Strip salience and novelty alone is a wandering signal; force greedy precision and you shut the wandering down, which helps.
If asked why all three ablations land at identical 47%: once you zero w_sal and w_nov AND collapse β→0.125, the agents become the same extrinsic-only policy by construction. That's exactly what the toggle is meant to show.
-->

---

# What each gap tells us

<div class="grid grid-cols-3 gap-4 pt-1 text-xs">

<div class="p-3 rounded-lg border-l-4 border-rose-400 bg-white/5">

### combined · −37pp
**Full EFE:** surveys the site, inspects top candidates at altitude, commits on the most-target-like.

**Ablated:** extrinsic-only + greedy β collapses to the shared baseline. No mechanism for deliberate information-gathering — commits are driven by passive sightings alone.

<div class="pt-1.5 opacity-75">Full cost of removing the two information-gathering terms.</div>

</div>

<div class="p-3 rounded-lg border-l-4 border-rose-400 bg-white/5">

### active_inference · −33pp
AI is **salience-only** — no novelty.

Salience is doing most of the heavy lifting: it targets viewpoints at cells whose class is still uncertain. That's the term that earns nearly all of the combined agent's gain.

<div class="pt-1.5 opacity-75">Novelty adds cross-episode transfer, but per-episode salience carries most of the win.</div>

</div>

<div class="p-3 rounded-lg border-l-4 border-amber-400 bg-white/5">

### active_learning · **+20pp ⚠**
AL is **novelty-only** — *worse* than its ablation.

Novelty rewards unvisited cells regardless of task-relevance. Without salience's targeting, it's a wandering signal. Strip it and force greedy precision → agent commits to best passive sighting, which *helps*.

<div class="pt-1.5 opacity-75">Lesson: information-gain has to be <i>aimed</i>. Curiosity without direction is worse than none.</div>

</div>

</div>

<div class="pt-4 text-sm">

**One-liner.** Curiosity earns the full 37pp above the extrinsic baseline — but only when aimed. Salience is what aims it. Combined more than doubles greedy (83% vs 40%). That's what the three-term decomposition buys on an actual unknown environment.

</div>

<!--
Key asymmetry: salience does most of the work (AI alone ≈ combined). Novelty alone is actively harmful on v2 because the environment is random each episode — there's nothing structural for bare novelty to exploit without salience to say which cells *matter*.
Controls reminder: greedy already had w_sal=w_nov=0 and β=0.125 by construction, so the toggle is a no-op. Random samples uniformly over actions; the toggle skips it. ±10pp RNG noise at 30 episodes.
-->

---
layout: center
class: text-center
---

# Drone work at Munin Dynamics
<div class="pt-2 text-sm opacity-70">Feb — Aug 2025 · volunteer engineer · camera-guided micro-interceptor</div>

<div class="pt-3 text-sm opacity-80 max-w-3xl mx-auto">

Munin's bet: the cheapest, smallest missile system that can down an FPV drone — pocket-sized, scalable to Ukraine-level threat volumes. I went in as a volunteer to build the perception stack on the interceptor itself.

</div>

<div class="grid grid-cols-3 gap-6 pt-6 text-left text-sm">

<div>

**Problem**
- Detect a small quadcopter out to **200 m**
- 150 m/s closing velocity
- SWaP: <30 mm diameter processing board
- $250 production BoM

</div>
<div>

**What I built**
- Camera / SBC selection against DRI / Johnson criteria
- FOMO detector, trained on-platform, int8-quantised
- Capture → detect → track on a Digi ConnectCore 93
- PyTorch + OpenCV + Docker; field-test support

</div>
<div>

**Numbers that mattered**
- FOMO vs YOLOv11n: **90% recall vs mAP50 0.27**
- Model size: **~200 KB** (vs ~5 MB)
- Inference: **<1 ms** at 128×128
- BoM well inside $250

</div>
</div>

<!--
If asked "why did you leave Munin?" — it's an excellent applied problem but the ceiling is a better detector; I want to work on the planner above it.
-->

---

# Before Munin · autonomous UAS at Frazer-Nash
<div class="pt-2 text-sm opacity-70">Sept 2022 – Apr 2024 · maritime defence · multirotor loitering platform · Python / C++ / ROS2 / Gazebo-Unity</div>

<div class="pt-3 text-sm opacity-80 max-w-4xl mx-auto">

A tactical multirotor analogous to <a href="https://www.stm.com.tr/en/kargu-autonomous-tactical-multi-rotor-attack-uav" target="_blank" class="underline decoration-dotted">STM's Kargu</a> — mini-class, EO/IR, loitering ISR plus precision strike — adapted for maritime defence. Onboard vision, target tracking and navigation on SWaP-constrained hardware, closing the loop in real time.

</div>

<div class="grid grid-cols-2 gap-8 pt-5 text-sm">

<div class="border-l-4 border-slate-400 pl-4">

**The system**

- Detector + tracker → control, tightly coupled
- Target estimates acted on **immediately** downstream
- Worked well when observations were clean
- I rebuilt the vision stack **20 → 70+ fps** on edge hardware while improving detection accuracy
- Led the technical design on a new **£235k bid**

</div>
<div class="border-l-4 border-amber-500 pl-4">

**Where it broke**

- Ambiguous scenes — clutter, sun-glint, partial occlusion
- **Confidence ≠ uncertainty**: the detector was sure when it shouldn't have been
- No explicit belief over the environment
- No mechanism to trade off exploration vs exploitation
- The gap wasn't perception or control — it was the **shared, uncertainty-aware layer** between them

</div>

</div>

<div class="pt-4 text-sm font-semibold opacity-90">
That missing layer is what drew me to active inference — and to this paper.
</div>

<!--
Keep this tight: the hook is the diagnosis, not the bid value. The £235k detail is there if asked "did you own scope as well as code?". Maritime detail lets me answer "what sort of targets?" without over-disclosing. If pressed on Kargu — I'm citing it as the nearest public analogue for the class of system, not the specific programme.
-->

---

# Where this work meets Stanhope

<div class="text-sm opacity-80 pt-1">

Two perception stacks, same ceiling. Both were discriminative layers; both broke when confidence stopped meaning *what the system knows*. The layer above them is what Stanhope is building.

</div>

<div class="grid grid-cols-2 gap-8 pt-2 text-sm">

<div class="border-l-4 border-slate-400 pl-4">

**What I shipped — twice**

- Pixels → $\{\text{target},\ \varnothing\}$ in milliseconds on SWaP-tight hardware
- No posterior — no notion of *"I don't know"*
- Retrain per domain: clutter, lighting, target drift
- Greedy downstream: chase the highest-confidence box

</div>
<div class="border-l-4 border-emerald-400 pl-4">

**What active inference *adds***

- Explicit $q(\theta)$ — a budget for what's unknown
- Salience + novelty planned *ahead* → **where should the sensor look next?**
- One objective drives pursuit *and* exploration — no $\varepsilon$-greedy
- Self-terminating curiosity (digamma, closed form)

</div>

</div>

<div class="pt-2 text-xs opacity-75">

Stanhope's Real World Model is pitched at exactly this layer — adaptive, uncertainty-aware behaviour on small autonomous platforms, trialled in drone & robotics partnerships. Frazer-Nash's SENTINEL (RL for sensor management) is the same problem shape; active inference derives it from one variational objective instead of a hand-engineered reward.

</div>

<div class="pt-2 text-sm font-semibold">
The offering: I've shipped the perception pipeline that sits <em>underneath</em> a Stanhope-style agent on a defence UAS — twice. What I came here for is the agent on top.
</div>

<!--
If asked about Frazer-Nash SENTINEL: it's an RL-for-sensor-management system; active inference collapses the same objective to salience + novelty without a separately learned reward model.
-->

---
layout: center
class: text-center
---

# Thanks

<div class="pt-8 text-base opacity-80">

Sim · <a href="https://jakeywatson.github.io/stanhope/sim/" target="_blank">jakeywatson.github.io/stanhope/sim</a><br/>
Source · <a href="https://github.com/jakeywatson/stanhope" target="_blank">github.com/jakeywatson/stanhope</a>

</div>

<div class="pt-12 text-2xl">
Questions.
</div>
