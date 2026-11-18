# TARA-FL interactive dashboard

A single-page, dependency-free dashboard with **two strictly separated data
sources**:

## Real Results tab (actual Python experiment metrics)

Renders only metrics exported by the real pipeline
(`experiments/run_unified.py` -> `experiments/results_export.py`): real
accuracy/loss measured on the MNIST test set, real per-client trust scores
from `TrustEngine`, real quarantine/zone transitions from the
`QuarantineManager`, real round-risk from `RoundRisk`, and real aggregation
decisions from the adaptive router. Every value traces to an experiment ID,
method, seed and round (see the Provenance panel).

### Mode A - Live local backend (round-by-round streaming)

Terminal 1 - start the backend (FastAPI + Server-Sent Events; falls back to
the Python standard library automatically):

```bash
python visualization/live_server.py --port 8000
```

Terminal 2 - run the real experiment:

```bash
python experiments/run_unified.py --method all --num_clients 10 --malicious_ratio 0.3 --attack gradient_scale --rounds 5 --seeds 42
```

Open <http://127.0.0.1:8000>, select the **Real Results** tab. The page
connects automatically and every completed round appears **without any manual
refresh** - accuracy/loss curves grow, trust lines update, quarantine
transitions stream in. When training finishes, the final metrics and timestamp
are shown; if the process stops early, all completed rounds are preserved; if
the backend is not running, the tab says so and shows **no** data (never
simulated numbers).

Windows CMD equivalents:

```bat
python visualization\live_server.py --port 8000
python experiments\run_unified.py --method all --num_clients 10 --malicious_ratio 0.3 --attack gradient_scale --rounds 5 --seeds 42
```

The experiment history dropdown (fed by `dashboard_data/index.json`) lets you
re-open any previous run; the live stream continues in the background.

### Mode B - Published results (GitHub Pages / static hosting)

GitHub Pages is static hosting: it cannot execute the Python backend or
connect to a visitor's machine. The public dashboard therefore shows the
**latest published actual experiment** as a clearly labeled static snapshot
(generation timestamp + experiment ID + config), plus the full published
history.

Publish the latest run:

```bash
python visualization/publish_results.py    # copies dashboard_data -> demo/results
git add demo/results
git commit -m "Publish latest real TARA-FL experiment results"
git push
```

`demo/results/latest.json`, `demo/results/index.json` and
`demo/results/history/` are served with the page; the tab fetches them
relative to its own URL. If nothing has been published yet, the tab shows
publish instructions instead of data.

## Simulation tab (synthetic illustration)

The original interactive demo of the trust mechanism -- five-stage trust
scoring, zone classification with quarantine/recovery, round-risk assessment
and the adaptive aggregation router -- on **synthetic** client updates, with
10 clients and 4 aggregation rules running side by side on the same updates:

- FedAvg (plain average)
- Trust+FedAvg (trust-weighted average)
- Trust+Robust (fixed trimmed mean)
- TARA-FL (adaptive router: FedAvg / trust-trim / coordinate median)

### Run it

No install, no build, no server needed. `index.html` is fully self-contained
(CSS, JavaScript and fonts are inlined -- it makes zero network requests):

- double-click `index.html`, or
- serve it with `python -m http.server 8000` in this folder and open
  <http://localhost:8000>

Works offline, from disk, or on GitHub Pages.

### What to try

1. Default settings (Label Flipping, 20%, non-IID, seed 42) -> Start. Watch the
   trust of the malicious clients (red lines) decay over the first rounds until
   they cross into quarantine, and the router switch away from plain FedAvg.
2. Set Attack = Gradient Scaling x10, Malicious = 30%, Data = IID, Rounds = 40.
   FedAvg collapses while TARA-FL finishes around 90% -- this mirrors the
   headline result in the report (22.65% vs 90.00% at 30% attackers).
3. Swap to FedAvg (Baseline) in Experiment Mode to make the stat cards show the
   baseline's numbers instead.
4. Same seed gives the same run, so the demo is repeatable when presenting.

### A note on the numbers

Client updates are synthetic 8-D vectors and the accuracy/loss trajectories
come from a small response model calibrated to the results in our report -- not
the original MNIST training loop (that is the full pipeline in this repo, see
`experiments/run_unified.py`). The trust/zone/risk/routing logic itself is the
real thing. The tab is labeled *synthetic illustration* and must not be used
as a source of experimental claims -- use the **Real Results** tab for that.

## Layout

One file: `index.html` contains everything (page, styles, live/static result
rendering, simulation engine, SVG charts, embedded fonts). Real experiment
documents live in `results/` (published) and `dashboard_data/` (generated
locally at the repo root by the exporter).
