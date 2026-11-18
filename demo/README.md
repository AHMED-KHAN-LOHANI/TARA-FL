# TARA-FL interactive demo

A single-page, dependency-free demo of the TARA-FL trust pipeline. It runs the
full mechanism described in the report -- five-stage trust scoring, zone
classification with quarantine/recovery, round-risk assessment and the adaptive
aggregation router -- on synthetic client updates, with 10 clients and 4
aggregation rules running side by side on the same updates:

- FedAvg (plain average)
- Trust+FedAvg (trust-weighted average)
- Trust+Robust (fixed trimmed mean)
- TARA-FL (adaptive router: FedAvg / trust-trim / coordinate median)

## Run it

No install, no build, no server needed. `index.html` is fully self-contained
(CSS, JavaScript and fonts are inlined -- it makes zero network requests):

- double-click `index.html`, or
- serve it with `python -m http.server 8000` in this folder and open
  <http://localhost:8000>

Works offline, from disk, or on GitHub Pages.

## What to try

1. Default settings (Label Flipping, 20%, non-IID, seed 42) -> Start. Watch the
   trust of the malicious clients (red lines) decay over the first rounds until
   they cross into quarantine, and the router switch away from plain FedAvg.
2. Set Attack = Gradient Scaling x10, Malicious = 30%, Data = IID, Rounds = 40.
   FedAvg collapses while TARA-FL finishes around 90% -- this mirrors the
   headline result in the report (22.65% vs 90.00% at 30% attackers).
3. Swap to FedAvg (Baseline) in Experiment Mode to make the stat cards show the
   baseline's numbers instead.
4. Same seed gives the same run, so the demo is repeatable when presenting.

## A note on the numbers

Client updates are synthetic 8-D vectors and the accuracy/loss trajectories
come from a small response model calibrated to the results in our report -- not
the original MNIST training loop (that is the full pipeline in this repo, see
`experiments/run_unified.py`). The trust/zone/risk/routing logic itself is the
real thing. The "Reported Results" table on the About tab shows the reference
numbers the model was fitted to.

## Layout

One file: `index.html` contains everything (page, styles, simulation engine,
SVG charts, embedded fonts). There is nothing else to install or download.
