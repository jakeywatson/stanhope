"""Generate distribution-shape visuals for the Schwartenbeck theory slides.

Each figure has a transparent-ish white background designed to sit inside a
white card on a dark Slidev slide. Colours are consistent across figures so a
reader can follow prior / likelihood / posterior semantics visually.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import digamma

OUT = Path(__file__).resolve().parents[1] / "public"
OUT.mkdir(parents=True, exist_ok=True)

# ─── palette ──────────────────────────────────────────────────────
C_PRIOR = "#94a3b8"       # slate-400 — what the agent believed
C_LIKELIHOOD = "#f59e0b"  # amber-500 — evidence
C_POSTERIOR = "#10b981"   # emerald-500 — updated belief
C_ACCENT = "#0284c7"      # sky-600
C_MUTE = "#6b7280"        # gray-500
C_KL_FILL = "#c084fc"     # violet-400 — info-gain shaded area

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#0f172a",
    "axes.titlecolor": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 160,
})


def save(fig, name: str):
    fig.tight_layout()
    out = OUT / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}")


# ─── slide 5 · A-matrix / risky-arm belief ─────────────────────────
def fig_beta_sharpening_basic():
    xs = np.linspace(0, 1, 400)
    configs = [
        ((1, 1), C_PRIOR, "Dir(1, 1)\nflat prior"),
        ((3, 1), C_ACCENT, "Dir(3, 1)\nafter 2 large"),
        ((7, 3), C_POSTERIOR, "Dir(7, 3)\nafter 6 large, 2 none"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    for i, ((a, b), col, lbl) in enumerate(configs):
        ys = beta_dist(a, b).pdf(xs)
        ax.plot(xs, ys, color=col, lw=2.3, label=lbl)
        mean = a / (a + b)
        ax.axvline(mean, color=col, lw=1, ls="--", alpha=0.6)
        # stagger vertically so Dir(3,1) and Dir(7,3) labels (0.75 vs 0.70) don't collide
        y_off = -0.30 - 0.28 * i
        ax.annotate(f"E[p]={mean:.2f}", xy=(mean, 0), xytext=(mean, y_off),
                    color=col, fontsize=9, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=-1.0)
    ax.set_xlabel("p — probability of reward from risky arm")
    ax.set_ylabel("density")
    ax.set_title("Belief over the risky column sharpens with each outcome")
    ax.legend(frameon=False, loc="upper center", ncols=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.22))
    save(fig, "beta-sharpening-basic.png")


# ─── slide 6 · inference = prior × likelihood → posterior ──────────
def fig_posterior_update():
    labels = ["start", "safe", "risky"]
    prior = np.array([0.2, 0.4, 0.4])
    like = np.array([0.10, 0.90, 0.20])     # likelihood of a noisy "small" obs
    unnorm = prior * like
    post = unnorm / unnorm.sum()
    panels = [("prior  q(s)", prior, C_PRIOR),
              ("likelihood  p(o | s)", like, C_LIKELIHOOD),
              ("posterior  q(s | o)", post, C_POSTERIOR)]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9),
                             gridspec_kw={"wspace": 0.55})
    for ax, (title, data, col) in zip(axes, panels):
        ax.bar(labels, data, color=col, width=0.7)
        for x, v in enumerate(data):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=9,
                    color="#0f172a")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.set_yticks([0, 0.5, 1.0])
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#cbd5e1")
    # operator glyphs
    fig.text(0.355, 0.50, "×", ha="center", va="center",
             fontsize=26, color=C_MUTE)
    fig.text(0.665, 0.50, "∝", ha="center", va="center",
             fontsize=26, color=C_MUTE)
    fig.suptitle("Bayesian state inference — the discrete analogue of predict × measure",
                 y=1.02, fontsize=11.5, color="#0f172a")
    save(fig, "posterior-update.png")


# ─── slide 8 · three-term panel ────────────────────────────────────
def fig_three_terms():
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.3),
                             gridspec_kw={"wspace": 0.38})

    # ── 1. extrinsic — log-preferences over observations ─────────
    ax = axes[0]
    obs = ["centre", "small", "large", "none"]
    c = np.array([0, 2, 4, -2])
    colors = [C_PRIOR if v < 0 else (C_POSTERIOR if v > 2 else C_ACCENT) for v in c]
    ax.bar(obs, c, color=colors, width=0.68)
    for x, v in enumerate(c):
        ax.text(x, v + (0.25 if v >= 0 else -0.5), f"{v:+d}",
                ha="center", fontsize=9, color="#0f172a")
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_title("1 · Extrinsic\n$\\log\\,p(o \\mid C)$", fontsize=10.5)
    ax.set_ylim(-3.5, 5)
    ax.set_ylabel("log-preference")
    ax.set_xticklabels(obs, fontsize=9)

    # ── 2. salience — KL between q(s) and q(s|o) ────────────────
    ax = axes[1]
    states = ["start", "safe", "risky"]
    q_prior = np.array([0.33, 0.33, 0.34])
    q_post = np.array([0.05, 0.80, 0.15])
    x = np.arange(len(states))
    w = 0.35
    ax.bar(x - w/2, q_prior, width=w, color=C_PRIOR, label="$q(s\\,|\\,\\pi)$")
    ax.bar(x + w/2, q_post,  width=w, color=C_POSTERIOR, label="$q(s\\,|\\,o,\\pi)$")
    # KL hatching overlay
    for xi, (a, b) in enumerate(zip(q_prior, q_post)):
        ax.fill_between([xi - w/2, xi + w/2], max(a, b), min(a, b),
                        color=C_KL_FILL, alpha=0.25, step=None)
    ax.set_xticks(x); ax.set_xticklabels(states, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("2 · Salience\n$D_{\\mathrm{KL}}[q(s|o)\\,\\|\\,q(s)]$", fontsize=10.5)
    ax.set_ylabel("probability")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # ── 3. novelty — KL between two Beta densities ─────────────
    ax = axes[2]
    xs = np.linspace(0, 1, 400)
    before = beta_dist(2, 2).pdf(xs)
    after = beta_dist(6, 3).pdf(xs)
    ax.plot(xs, before, color=C_PRIOR, lw=2, label="$q(\\theta)$")
    ax.plot(xs, after, color=C_POSTERIOR, lw=2, label="$q(\\theta\\,|\\,s,o)$")
    ax.fill_between(xs, before, after,
                    where=(after > before), color=C_KL_FILL, alpha=0.25)
    ax.fill_between(xs, before, after,
                    where=(after <= before), color=C_KL_FILL, alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title("3 · Novelty\n$D_{\\mathrm{KL}}[q(\\theta|s,o)\\,\\|\\,q(\\theta)]$", fontsize=10.5)
    ax.set_xlabel("parameter  θ")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    save(fig, "three-terms-panel.png")


# ─── slide 9 · Dirichlet moments / digamma ─────────────────────────
def fig_dirichlet_digamma():
    xs = np.linspace(0.001, 0.999, 400)
    steps = [(1, 1), (2, 3), (5, 8), (15, 20)]
    cmap = ["#94a3b8", "#38bdf8", "#0284c7", "#10b981"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.2),
                                  gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.32})
    for (a, b), col in zip(steps, cmap):
        ys = beta_dist(a, b).pdf(xs)
        ax.plot(xs, ys, color=col, lw=2.1,
                label=f"α=({a},{b})  E[θ]={a/(a+b):.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("parameter θ"); ax.set_ylabel("density")
    ax.set_title("Posterior sharpens as counts grow (conjugacy)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # digamma log-moment
    total = np.arange(2, 41)
    psi_diff = digamma(total / 2) - digamma(total)  # ψ(α_i) − ψ(α_0) for balanced counts
    ax2.plot(total, psi_diff, color=C_ACCENT, lw=2.1)
    ax2.set_xlabel("$\\alpha_0 = \\sum_i \\alpha_i$")
    ax2.set_ylabel("$\\mathbb{E}[\\ln\\theta_i] = \\psi(\\alpha_i) - \\psi(\\alpha_0)$", fontsize=10)
    ax2.set_title("Log-moment is closed-form")
    ax2.set_ylim(-1.35, -0.55)
    ax2.axhline(np.log(0.5), color=C_MUTE, ls="--", lw=1, alpha=0.6)
    ax2.text(38, np.log(0.5) - 0.04, "ln 0.5",
             fontsize=8.5, color=C_MUTE, ha="right", va="top")
    save(fig, "dirichlet-digamma.png")


# ─── slide 10 · policy softmax temperature ─────────────────────────
def fig_softmax_temperature():
    G = np.array([2.5, 1.8, 1.2, 0.4])       # EFE per policy (lower = preferred)
    neg_G = -G
    policies = [f"π{i+1}" for i in range(len(G))]
    betas = [(0.125, "β = 2⁻³  near-argmin"),
             (1.0,   "β = 1  default"),
             (8.0,   "β = 2³  near-uniform")]
    colors = [C_ACCENT, C_POSTERIOR, C_LIKELIHOOD]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8),
                             gridspec_kw={"wspace": 0.35})
    for ax, (beta_val, title), col in zip(axes, betas, colors):
        scores = neg_G / beta_val
        scores -= scores.max()
        probs = np.exp(scores); probs /= probs.sum()
        ax.bar(policies, probs, color=col, width=0.68)
        for x, v in enumerate(probs):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9,
                    color="#0f172a")
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=10.5)
        ax.set_yticks([0, 0.5, 1.0])
    fig.suptitle("$q(\\pi) = \\sigma(-G(\\pi)/\\beta)$  with fixed EFE values  $G=(2.5, 1.8, 1.2, 0.4)$",
                 y=1.02, fontsize=10.5, color="#0f172a")
    save(fig, "softmax-temperature.png")


if __name__ == "__main__":
    fig_beta_sharpening_basic()
    fig_posterior_update()
    fig_three_terms()
    fig_dirichlet_digamma()
    fig_softmax_temperature()
