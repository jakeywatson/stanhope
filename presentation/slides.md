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
layout: two-cols
---

# The T-maze task

<v-clicks>

- Agent starts at centre
- Left arm: **safe** — small reward, deterministic
- Right arm: **risky** — large reward with unknown probability $p$
- One trial = one arm choice
- $N$ trials to maximise total reward

</v-clicks>

<div v-click class="pt-6 text-sm opacity-80">
The agent does not know $p$. It must choose between exploiting what it knows (safe) and learning what it doesn't (risky) — without an explicit exploration bonus.
</div>

::right::

<div class="pl-6">

```mermaid {scale: 0.8}
graph TD
  S[Start] -->|a1| Safe
  S -->|a2| Risky
  Safe[Safe arm<br/>+small reward<br/>always]
  Risky[Risky arm<br/>+large w.p. p<br/>-no-reward w.p. 1-p]
```

<div class="pt-4 text-sm opacity-75">
<b>Why this task?</b> It isolates the learning problem. No hidden context, no partial observability — the only uncertainty is about $p$, a parameter of the observation model.
</div>

</div>

---

# Figure 2 — the generative model

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

# The A-matrix (observation model)

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

<div class="pt-4">

Two deterministic columns (start, safe). One **unknown** column (risky), parametrised by a single scalar $p$.

The agent represents its belief about this column as a **Dirichlet distribution** with concentration parameters $\boldsymbol{\alpha} \in \mathbb{R}^{2}_+$ over $\{\text{large}, \text{none}\}$:

$$
q(A_{\cdot,\text{risky}}) = \text{Dir}(\boldsymbol{\alpha}), \qquad
\mathbb{E}[A_{o,\text{risky}}] = \frac{\alpha_o}{\alpha_0}, \ \ \alpha_0 = \textstyle\sum_o \alpha_o
$$

The paper starts at $\boldsymbol{\alpha} = [1,1]$ — maximum a priori uncertainty about $p$.

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

<div class="pt-4">

- $\sigma(\cdot)$ is the softmax — it falls out of the KKT conditions for constrained minimisation over the simplex
- $\ln A^{\!\top} \tilde o$ has a principled interpretation: **log-evidence** arriving at state $s$
- This is the forward-backward algorithm in active-inference clothing

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
The minus signs matter: minimising $G$ means <b>low expected cost</b> <i>and</i> <b>high expected information gain</b>. Exploration is not a bonus grafted on — it drops out of the same objective as reward-seeking.
</div>

---

# The three terms · what each one <em>buys</em> you

<div class="grid grid-cols-3 gap-6 pt-2 text-sm">

<div>

### 1 · Extrinsic
$$-\mathbb{E}_{q(o\mid\pi)}\!\left[\ln p(o \mid C)\right]$$

"Do I expect to see preferred outcomes?"

<div class="pt-2">
Drives **exploitation**. Purely reward-seeking — this is the only term a greedy RL agent optimises.
</div>
</div>

<div>

### 2 · Salience
$$-\mathbb{E}_{q}[\,D_{\text{KL}}\!\left[q(s\mid o)\,\|\,q(s)\right]\,]$$

"How much does this observation shift my beliefs about hidden state?"

<div class="pt-2">
Drives **hidden-state exploration**. Useful when the task has a cue revealing context — drives the agent toward informative observations.
</div>
</div>

<div>

### 3 · Novelty
$$-\mathbb{E}_{q}[\,D_{\text{KL}}\!\left[q(\theta\mid s,o)\,\|\,q(\theta)\right]\,]$$

"How much does this observation teach me about model parameters?"

<div class="pt-2">
Drives **parameter exploration**. Collapses toward zero as beliefs about $\theta$ concentrate — self-terminating curiosity.
</div>
</div>

</div>

<div class="pt-6 text-sm opacity-75">
For the bare T-maze (no cue), salience vanishes — $s$ is deterministic given $a$. Novelty is the whole exploration engine.
</div>

---

# Parameter learning · why Dirichlet

The posterior over the risky column of $A$ is categorical-likelihood × Dirichlet-prior:

$$
p(\boldsymbol\theta \mid \mathcal{D}) \;\propto\; \text{Cat}(\mathcal{D}\mid\boldsymbol\theta)\,\text{Dir}(\boldsymbol\theta \mid \boldsymbol\alpha)
\;=\; \text{Dir}\!\left(\boldsymbol\alpha + \mathbf{n}\right)
$$

— **conjugate**. The update rule is just: observe $o$ in state $s$, then $\alpha_{o,s} \mathrel{+}= 1$.

<div class="pt-4">

Two moments we actually need:

$$
\mathbb{E}[\theta_i] = \frac{\alpha_i}{\alpha_0}
\qquad
\mathbb{E}[\ln \theta_i] = \psi(\alpha_i) - \psi(\alpha_0)
$$

where $\psi$ is the digamma function and $\alpha_0 = \sum_i \alpha_i$.

The second identity is what makes the **novelty term** computable in closed form — without sampling, without a learned amortiser. You evaluate digammas at integer counts. That's it.

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

$\gamma$ is **precision** over policies. In the paper it's expected under a Gamma prior with rate $\beta$:

<div class="pt-2 text-sm">

| $\beta$ | Behaviour | Mapping to RL |
|---|---|---|
| $2^{-3}$ | high precision, near-argmin | greedy / low-temperature softmax |
| $1$ | standard | usual temperature |
| $2^{3}$ | low precision, nearly uniform | high-temperature / random |

</div>

<div class="pt-4">

**Why a prior on $\gamma$ at all?** It lets the agent *modulate its own confidence*. When $G(\pi)$ values are close, $\gamma$ naturally shrinks — "I genuinely don't know which is better, so don't commit."

This is the lever that separates the paper's four experimental regimes.

</div>

---

# Implementation choices

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
layout: center
---

# Live reproduction

<div class="pt-4 pb-4">

Python (NumPy) model running in the browser via **Pyodide**.<br/>
Same `.py` files run locally for validation; the sim matches paper outputs.

</div>

<iframe
  src="https://jakeywatson.github.io/stanhope/sim/"
  class="w-full rounded-lg shadow-xl"
  style="height: 60vh; border: 1px solid rgba(255,255,255,0.1);"
></iframe>

<div class="absolute bottom-4 right-8 text-xs opacity-60">
Switch agent in the top-right · controls in the side panel
</div>

<!--
Demo points to hit live:
1. Active-learning agent: show early risky-arm sampling, novelty term high
2. Watch concentration parameters grow, novelty decay
3. Switch to random (β=2³): exploration never decays
4. Switch to greedy: never discovers risky arm even when it's good
5. Resist live-tweaking — stick to the script
-->

---

# The key result · Figure 6

<div class="grid grid-cols-2 gap-8">

<div>

**What we should see**

<v-clicks>

- Active-learning agent: probability of risky choice **decays** over trials as uncertainty resolves
- Random-exploration agent: probability of risky choice is **flat** — no self-termination
- Greedy agent: if initial estimate $\mathbb{E}[p]$ is low, never recovers — risky column is never revisited

</v-clicks>

</div>
<div>

**What this tells us**

<v-clicks>

- Exploration collapses *from within the same objective* that drives exploitation
- No separate exploration schedule (ε-greedy, entropy bonus) needed
- The decay rate is a closed-form function of $\psi(\alpha_0)$ — you can compute when curiosity will "turn off"

</v-clicks>

</div>

</div>

<div v-click class="pt-8 text-center text-lg opacity-90">
This is the paper's strongest claim — and it replicates.
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
layout: two-cols
---

# Extension 1 · grid-maze

- 10×10 grid, fog-of-war observation model
- Goal hidden at unknown cell
- Agent's $A$ is a learned map: $P(\text{wall}, \text{open}, \text{goal} \mid s)$
- Salience now *matters* — visiting a cell reduces state-entropy about its contents
- Novelty still fires for uncertain cells

<div class="pt-4 text-sm opacity-80">

Shows the same mechanism extends to **real spatial POMDPs**, not just the 3-state toy.

</div>

::right::

<div class="pl-6">

**Why this matters for Stanhope**

Spatial navigation under partial observability is the default, not the exception. The T-maze version of the paper is didactic — the production question is: does the decomposition still give you exploration-that-terminates when the state space is non-trivial?

In the sim: yes, qualitatively. The novelty term decays, coverage increases, path cost is bounded. But — see prior slide — the combinatorics start biting at $\tau > 2$.

</div>

---
layout: two-cols
---

# Extension 2 · drone search

- 3D grid world with obstacles
- Drone has a camera frustum — partial FoV
- Targets hidden in occluded cells
- Belief over (target-present, target-absent) per cell

**Policy**: where to fly *and* where to look

<div class="pt-4 text-sm opacity-80">

Salience term chooses viewpoints that *maximally reduce uncertainty* over target locations. This is essentially **next-best-view planning** with a free-energy objective.

</div>

::right::

<div class="pl-6">

**Why this matters**

This is closer to what a perception-led autonomy system actually does. It reframes exploration as *information gathering under embodiment constraints* rather than as a schedule.

The caveat is honest: I built this as a didactic extension in a discrete world. The combinatorial blow-up from the prior slide is real — continuous motion + high-dim observations need amortised policy posteriors, not exhaustive EFE sums.

</div>

---
layout: center
class: text-center
---

# Drone work at Munin Dynamics
<div class="pt-2 text-sm opacity-70">Feb — Aug 2025 · camera-guided interceptor (Micro-MANPAD)</div>

<div class="grid grid-cols-3 gap-6 pt-8 text-left text-sm">

<div>

**Problem**
- Detect 7″ quadcopter at 30–350 m
- 150 m/s closing velocity
- SWAP: <30 mm diameter processing board
- $250 production BoM

</div>
<div>

**What I built**
- Camera / SBC hardware selection against DRI/Johnson criteria
- FOMO detector trained on-platform, quantised for edge
- Full capture → detect → track pipeline on a Digi ConnectCore 93
- Field-test support

</div>
<div>

**Numbers that mattered**
- FOMO beat YOLOv11n: **90% vs mAP50 0.27**
- Model size: **~200 KB** (vs ~5 MB for YOLO)
- Inference: **<1 ms** at 128×128 on the test SBC
- Total BoM: well inside $250

</div>
</div>

<div class="pt-6 text-sm opacity-80">
Earlier at Frazer-Nash (autonomous vehicle vision): rebuilt the perception pipeline from 20 → 70+ fps on Jetson edge hardware while improving detection accuracy.
</div>

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
