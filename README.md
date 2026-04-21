# Stanhope interview · Schwartenbeck et al. (2019)

A talk + interactive simulator built around *Computational mechanisms of curiosity and goal-directed exploration* (Schwartenbeck, Passecker, Hauser, FitzGerald, Kronbichler, Friston · *eLife* 2019).

**Live links**
- Slides — <https://jakeywatson.github.io/stanhope/>
- Sim — <https://jakeywatson.github.io/stanhope/sim/>

The sim runs the paper's agent in NumPy, compiled to the browser via Pyodide, alongside three extensions that stress-test the same three-term expected-free-energy decomposition on progressively harder tasks.

## What's in here

```
presentation/       Slidev deck + speaker script
  slides.md           the deck itself
  SCRIPT.md           condensed bullet script (1–2 pages A4)
  public/             figures used in slides
sim/                Python model + Three.js frontend
  model/              NumPy implementation of Schwartenbeck + extensions
    scenarios/          tmaze · grid_maze · drone_search · drone_search_v2
  src/                TypeScript UI (Vite + Three.js)
    scenarios/          one scene controller per scenario
  scripts/            CLI tests (Phase smoke + Playwright visual)
.github/workflows/  GitHub Pages deploy (deck + sim together)
```

## Scenarios

| ID | Paper match | Adds |
|---|---|---|
| `tmaze` | Reproduces Figures 3 and 6 | — |
| `grid_maze` | 10×10 fog-of-war | Salience over cells |
| `drone_search` | 3D drone + camera frustum | Learned sensor model per altitude |
| `drone_search_v2` | Unknown 15×15 site, buildings, decoys | **Transferable Dirichlet prior across episodes** |

`grid_maze` and the v1 drone are hidden in the interactive UI by default. Use `?scenarios=all` to expose them.

## Running locally

```bash
# Deck (http://localhost:3030)
cd presentation && npm install && npm run dev

# Sim (http://localhost:5173)
cd sim && npm install && npm run dev
```

Pyodide downloads on first load and then caches — give the splash screen ~30 s.

## The ablation, in one paragraph

`runner.benchmark_batch(..., force_extrinsic_only=True)` zeroes each agent's `w_salience` and `w_novelty` and clamps β to 0.125. This reduces every agent to greedy-on-extrinsic alone. On the v2 unknown-site task (30 episodes/agent, 400-step cap): combined falls from 83% → 47%, active-inference from 80% → 47%, active-learning *rises* from 27% → 47%. The last is the interesting one — novelty without salience is a diffuse signal, and forcing a greedy commit-to-best helps by shutting the roaming down. Curiosity earns the gap above the extrinsic baseline, but only when it's aimed.

## Source

The model files in `sim/model/` follow the notation of Schwartenbeck et al. 2019 closely. The EFE decomposition (`free_energy.py`) and Dirichlet learning (`dirichlet.py`) are the ones to read first.
