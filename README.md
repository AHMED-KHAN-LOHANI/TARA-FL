# TARA-FL: Trust-Aware Robust Adaptive Federated Learning

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/tests-117%20passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)](https://farooqpersonal12.github.io/TARA-FL/demo/)

**TARA-FL** is a modular, research-grade framework for Byzantine-robust Federated Learning. It unifies **Dynamic Continuous Trust Scoring**, **Multi-Zone Quarantine/Eligibility Management**, **Multi-Signal Round-Risk Assessment**, and **Trust-Weighted Adaptive Aggregation** into an end-to-end defensible federated training loop.

## Live Demo & Real Experiment Dashboard

The interactive dashboard in `demo/` (a single self-contained HTML file) has two
data sources, kept strictly separate:

- **Real Results tab** — *actual* metrics produced by the Python pipeline
  (`experiments/run_unified.py`): real accuracy/loss from `server.evaluate()`,
  real trust scores from `TrustEngine`, real quarantine transitions from the
  `QuarantineManager`, real round-risk and real aggregation decisions.
  Two operating modes:
  - **Mode A — Live local backend** (`python visualization/live_server.py`):
    the browser connects over Server-Sent Events and every completed
    communication round appears automatically, without refresh, while training
    runs. If the backend is stopped, completed rounds remain visible and the
    dashboard says so; if no backend is running, no metrics are shown (never
    simulated ones).
  - **Mode B — Published results** (GitHub Pages): the latest *actual* run is
    exported to `demo/results/` and displayed as a static snapshot, clearly
    labeled with its experiment ID and timestamp. GitHub Pages is static
    hosting — it cannot run the Python backend or connect to your machine;
    the page states this instead of pretending to be live.
- **Simulation tab** — the original illustrative synthetic demo (calibrated
  response model). It is now explicitly labeled as an illustration; it is never
  mixed with real results.

**[Launch the TARA-FL Dashboard](https://farooqpersonal12.github.io/TARA-FL/demo/)**

Quick start (Windows CMD) — see section 6.5 for the full workflow:

```bat
cd C:\path\to\TARA-FL
pip install -r requirements.txt
python visualization\live_server.py          REM terminal 1: dashboard + SSE backend
python experiments\run_unified.py --method all --num_clients 10 --malicious_ratio 0.3 --attack gradient_scale --rounds 5 --seeds 42   REM terminal 2: real training
```

Open `http://127.0.0.1:8000`, switch to the **Real Results** tab, and watch the
real experiment stream in round by round.

---

## 1. End-to-End TARA-FL Architecture & Pipeline

```
                              ┌───────────────────────────────┐
                              │     Global Model (Round t)    │
                              └───────────────┬───────────────┘
                                              │ Broadcast
                                              ▼
                              ┌───────────────────────────────┐
                              │      Client Participation     │
                              └───────────────┬───────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │    Local SGD Client Training  │
                              └───────────────┬───────────────┘
                                              │ Parameter Updates (ΔW_k)
                                              ▼
                              ┌───────────────────────────────┐
                              │  Multi-Metric / PID Detection │
                              └───────────────┬───────────────┘
                                              │ Anomaly Scores & Distances
                                              ▼
                              ┌───────────────────────────────┐
                              │     Dynamic Trust Engine      │
                              │   - Peer-Relative MAD Scale   │
                              │   - Current Trust Decay       │
                              │   - Historical EMA Memory     │
                              │   - Temporal Persistence Cost │
                              └───────────────┬───────────────┘
                                              │ Continuous Trust T_k ∈ [0, 1]
                                              ▼
                              ┌───────────────────────────────┐
                              │    Trust Zone Classification  │
                              │   - Safe Zone (≥ 0.75)        │
                              │   - Probation [0.40, 0.75)    │
                              │   - Quarantine (< 0.40)       │
                              └───────────────┬───────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │       QuarantineManager       │
                              │   - Evict Quarantined Clients │
                              │   - Track Multi-Round Recovery│
                              └───────────────┬───────────────┘
                                              │ Eligible Clients S_t
                                              ▼
                              ┌───────────────────────────────┐
                              │     Round Risk Assessment     │
                              │   - Update Distances (D_k)    │
                              │   - Trust Deficit & Spread    │
                              │   - Suspicious Ratio (η_susp) │
                              │   - Validation Degradation    │
                              │   - Threat Type Categorization│
                              └───────────────┬───────────────┘
                                              │ Composite Risk (Low/Med/High)
                                              ▼
                              ┌───────────────────────────────┐
                              │      Adaptive Aggregation     │
                              │   - Low:   Trust FedAvg       │
                              │   - Med:   Trust Robust Trim  │
                              │   - High:  Trust Median       │
                              └───────────────┬───────────────┘
                                              │ Aggregated Model W_{t+1}
                                              ▼
                              ┌───────────────────────────────┐
                              │    Global Model Evaluation    │
                              │       & Next Round (t+1)      │
                              └───────────────────────────────┘
```

---

## 2. Core Methodological Modules

### A. Dynamic Trust Engine (`trustengine/`)
For each client $k$, trust $T_k^{(t)} \in [0, 1]$ is computed through a 5-stage pipeline:
1. **Peer-Relative MAD Anomaly**: Absolute update distance/PID score is normalized relative to population median and Median Absolute Deviation ($\text{MAD}$):
   $$\tilde{A}_k = 1.0 + \max\left(0, \frac{S_k - \text{Median}(S)}{\max(\text{MAD}(S), \epsilon)}\right)$$
2. **Current Trust**: Inverse quadratic degradation for anomalous deviation:
   $$T_{\text{curr}, k} = \frac{1}{1 + \max(0, \tilde{A}_k - 1 - \delta_{\text{normal}})^2}$$
3. **Historical Trust Memory**: Exponential Moving Average across preceding rounds:
   $$T_{\text{hist}, k} = \alpha T_k^{(t-1)} + (1-\alpha) T_{\text{curr}, k}$$
4. **Persistence Penalty**: Temporal accumulation of repeated adversarial behavior across rounds:
   $$P_k = \frac{\sum_{\tau=1}^t \gamma^{t-\tau} \text{Severity}_k^{(\tau)}}{\sum_{\tau=1}^t \gamma^{t-\tau}}$$
5. **Composite Trust Score**:
   $$T_k^{(t)} = w_{\text{hist}} T_{\text{hist}, k} + w_{\text{curr}} T_{\text{curr}, k} - w_{\text{pers}} P_k$$

### B. Trust Zones & QuarantineManager (`trustengine/quarantine.py`)
Clients are classified into three operating zones based on trust score $T_k$:
- **Safe Zone ($T_k \ge 0.75$)**: Client update is aggregated with full trust weighting.
- **Probation Zone ($0.40 \le T_k < 0.75$)**: Client update is down-weighted; client monitored for consecutive clean rounds.
- **Quarantine Zone ($T_k < 0.40$)**: Client update is **completely excluded** from model aggregation ($w_k = 0$). Quarantined clients must demonstrate $N_{\text{clean}} \ge 2$ consecutive clean rounds before graduating back to probation.

### C. Multi-Signal Round Risk Assessment (`risk/`)
Server computes a continuous composite risk score $R \in [0, 1]$ aggregating 5 orthogonal signals:
1. **Update Distance Anomaly** ($w=0.25$): Ratio of maximum client distance to median distance.
2. **Trust Deficit** ($w=0.25$): $1.0 - \text{Mean}(T_k)$.
3. **Suspicious Client Proportion** ($w=0.20$): Fraction of clients with $T_k < 0.75$.
4. **Worst-Client Deficit** ($w=0.15$): $1.0 - \min(T_k)$.
5. **Validation Degradation** ($w=0.15$): Accuracy drop relative to previous round baseline.

The composite risk score determines the **Risk Level**:
- **LOW** ($R < 0.30$): Clean or low-noise environment.
- **MEDIUM** ($0.30 \le R < 0.60$): Moderate poisoning or non-IID divergence.
- **HIGH** ($R \ge 0.60$): Severe coordinated adversarial attack.

### D. Adaptive Aggregation Strategy Routing (`aggregation/`)
Aggregation automatically adapts to the round risk level:
- **Low Risk $\to$ Trust-Aware FedAvg**:
  $$W_{t+1} = \sum_{k \in S_t} \frac{n_k T_k}{\sum_{j} n_j T_j} W_{t+1, k}$$
- **Medium Risk $\to$ Trust-Weighted Robust Trimming**:
  Trims top $\beta$-fraction of updates farthest from coordinate medians, weighted by $T_k$.
- **High Risk $\to$ Trust-Weighted Coordinate Median**:
  Computes element-wise median with trust-mass weighting, providing maximal breakdown point ($50\%$).
- **All Quarantined $\to$ Model Retention**:
  If all participating clients are quarantined, the server preserves current global weights $W_{t+1} = W_t$ without corrupting parameters.

---

## 3. Implemented Attacks

The repository contains modular, extensible attack implementations in [`attacks/`](attacks/):
1. **Clean Baseline (`none`)**: Standard benign federated training.
2. **Label Flipping (`label_flip`)**: Data poisoning where Byzantine clients train on flipped label mappings (e.g. $75\%$ noise, class $y \to (9 - y)$).
3. **Gradient Scaling (`gradient_scale`)**: Model poisoning where Byzantine clients scale parameter updates by a factor $\gamma$ (e.g. $\times 10.0$).
4. **Sign Flipping (`sign_flip`)**: Direction inversion where Byzantine clients submit $-1.0 \times \Delta W_k$, reversing the loss gradient while preserving norm magnitude.

---

## 4. Implemented Baseline & Comparative Methods

1. **Standard FedAvg**: Classical sample-size weighted averaging without defense.
2. **Trust + FedAvg**: Dynamic continuous trust weighting without robust coordinate trimming or quarantine.
3. **Trust + Fixed Robust Aggregation**: Dynamic trust weighting with fixed coordinate-wise robust trimming regardless of round risk.
4. **Full TARA-FL**: Continuous Trust Engine + QuarantineManager + Multi-Signal RoundRisk + Adaptive Aggregation.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch $\ge 2.0.0$

### Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-org/TARA-FL.git
cd TARA-FL

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Running Experiments & Reproduction

### Quick Demonstration Run
Run an end-to-end 5-round TARA-FL training execution with 3 clients and 1 malicious label-flip client:
```bash
python main.py
```

### Controlled 4-Way Comparative Experiment
Execute a controlled comparison across all 4 methods (`fedavg`, `trust_fedavg`, `trust_robust`, `tara`) under identical seeds, models, and data splits:
```bash
# 20% Gradient Scaling Attack under IID partition (10 clients, 5 rounds)
python experiments/run_unified.py --method all --num_clients 10 --malicious_ratio 0.2 --attack gradient_scale --scale_factor 10.0 --data_split iid --rounds 5 --seeds 42

# 20% Sign Flipping Attack under Non-IID Dirichlet (α=0.5) partition
python experiments/run_unified.py --method all --num_clients 10 --malicious_ratio 0.2 --attack sign_flip --data_split noniid --alpha 0.5 --rounds 5 --seeds 42
```

### Complete 20-Scenario Evaluation Matrix
Execute the complete matrix across 4 attack categories, 4 Byzantine proportions (0%, 10%, 20%, 30%), 2 data distributions (IID and Non-IID), and multiple random seeds:
```bash
python experiments/run_final_matrix.py --rounds 5 --seeds 42 43
```

### Statistical Synthesis & Figure Generation
Analyze all completed experimental trajectories and regenerate the 8 publication-grade visualization figures:
```bash
python experiments/analyze_matrix_results.py
```

### Interactive Real-Time Dashboard
Launch the built-in HTTP telemetry dashboard:
```bash
python visualization/dashboard.py --port 8080 --data experiments/matrix_raw
```
Navigate to `http://127.0.0.1:8080` in your web browser.

### Real Experiment Dashboard (Python results in the browser)

`demo/index.html` is a single self-contained file (styles, scripts and fonts
inlined) with **two data sources**:

- **Real Results tab** — renders *only* metrics exported by the actual Python
  pipeline. Nothing in this tab is simulated, estimated or extrapolated.
- **Simulation tab** — the original illustrative demo with a calibrated
  response model, now clearly labeled *synthetic illustration*. It is never
  mixed with real results.

The Python side is bridged by three small modules:

| Module | Role |
| --- | --- |
| `experiments/results_export.py` | Non-invasive exporter: after **every completed round** it atomically publishes `dashboard_data/live.json`, `dashboard_data/history/<experiment_id>.json` and `dashboard_data/index.json`. It captures the accuracy/loss already measured by `server.evaluate()`, the full per-client `TrustEngine.calculate_trust()` record (trust, PID score, relative anomaly, zone, quarantine flag), `RoundRisk` breakdowns, detector distances/PID scores, quarantine/zone transitions (diffed between rounds) and the adaptive aggregation decision. |
| `visualization/live_server.py` | FastAPI (+ uvicorn) backend serving the dashboard at `/` with REST endpoints (`/api/status`, `/api/live`, `/api/experiments`, `/api/experiments/<id>`) and an SSE stream at `/api/stream` that pushes a new `update` event within ~1 second of every completed round. Falls back to the Python standard library if FastAPI is unavailable. The backend never generates metrics. |
| `visualization/publish_results.py` | Copies the latest real run into `demo/results/` (`latest.json`, `history/`, `index.json`) so static hosting (GitHub Pages) can serve it. |

#### Mode A — live local execution (real-time, round-by-round)

Terminal 1 (backend):
```bash
python visualization/live_server.py --port 8000
```
Terminal 2 (real training):
```bash
python experiments/run_unified.py --method all --num_clients 10 --malicious_ratio 0.3 --attack gradient_scale --rounds 5 --seeds 42
```
Open `http://127.0.0.1:8000` → **Real Results** tab. The page connects via
`EventSource` and updates automatically after every completed round: accuracy
and loss curves grow round by round, trust trajectories, zone chips, risk
levels, quarantine transitions and aggregation decisions all reflect the real
engine state. No manual refresh is needed. Requirements 1-3, 7-10 of a fully
dynamic pipeline are covered:

- the dashboard detects a running experiment as soon as it starts (the
  exporter publishes an initial `running` snapshot);
- every metric is traceable to its experiment ID, method, seed and round
  (shown in the Provenance panel);
- when training completes, the real final metrics are shown with timestamp;
- if the process dies, all completed rounds are already on disk and remain
  viewable (status `stopped`/`failed` in the banner);
- if the backend is unreachable, the tab shows an explicit offline banner with
  the exact start command - it never substitutes synthetic data.

Works on any OS; on Windows run `python visualization\live_server.py`.

#### Mode B — published results on GitHub Pages (static)

GitHub Pages cannot execute Python or reach your locally running backend, so
the public dashboard shows the latest *published actual* experiment:

```bash
python visualization/publish_results.py    # dashboard_data -> demo/results/
git add demo/results
git commit -m "Publish latest real TARA-FL experiment results"
git push
```

The Real Results tab in "Published Results" mode loads `results/latest.json`
relative to the page, lists all published experiments from
`results/index.json`, and labels the view *"static snapshot - not a live
stream"* with the generation timestamp and experiment ID. In Live mode on
GitHub Pages, the page explains the limitation instead of attempting an
impossible connection.

#### Real experiment data format (JSON schema v1.1)

Each export document contains: `experiment_id`, `status`
(`running|completed|stopped|failed`), timestamps, full reproducible `config`
(dataset, clients, rounds, local epochs, attack, flip/scale factors,
malicious ratio **and the exact malicious client IDs**, split, Dirichlet
alpha, detector type, seeds), `environment` (Python/PyTorch versions),
`source` (script + argv + cwd), and per method → per seed → per round:
accuracy, loss, risk score/level + component breakdown, threat type,
selected aggregator, suspicious-client count, malicious weight share, round
wall time, per-client trust/PID/anomaly/zone/quarantine/eligibility,
`quarantine_transitions`, and detector `distances`/`pid_scores`. Methods that
do not compute a signal (e.g. FedAvg computes no trust scores) store `null`
explicitly instead of a fabricated value.

#### Reproducing the published experiment

The committed `demo/results/latest.json` was produced by:

```bash
python experiments/run_unified.py --method all --num_clients 10 --malicious_ratio 0.3 --attack gradient_scale --scale_factor 10.0 --data_split iid --rounds 5 --seeds 42
```

Its headline outcome (real measured values): plain **FedAvg collapses to
25.36%** final accuracy while **TARA-FL maintains 89.57%**, all 3 malicious
clients are quarantined by the `QuarantineManager` during training, and the
adaptive router leaves plain FedAvg after round 1. The exact JSON - every
round, every client - is in `demo/results/`.

### Interactive Browser Demo (synthetic illustration, no setup)

The **Simulation** tab of `demo/index.html` implements the full TARA-FL
defense loop on synthetic 8-D client updates with all 4 aggregation rules
computed side by side - useful to explain the mechanism interactively. It runs
entirely offline (`demo/` makes zero network requests):

```bash
cd demo
python -m http.server 8000    # optional; opening demo/index.html directly also works
```

The simulation is seeded, so the same configuration always reproduces the same
run. Suggested headline configuration: `Gradient Scaling x10`, `30% malicious`,
`IID`, `40 rounds` - matching the reported result in section 9 (22.65% vs
90.00%). Remember: this tab is a calibrated illustration; the **Real Results**
tab is the source for experimental claims.

---

## 7. Running Tests

Run the complete 117-test suite (92 original unit/integration tests + 25 new
dashboard-export & live-backend integration tests):
```bash
python -m pytest -v
```
The new tests (`tests/test_results_export.py`, `tests/test_live_server.py`)
verify: schema completeness, progressive per-round publishing, crash safety
(completed rounds preserved without finalization), honest `null` handling for
signals a method does not compute, quarantine-transition detection, the
history index, every REST endpoint, and real-socket SSE streaming that pushes
an `update` frame when a new round completes while the dashboard is connected.

---

## 8. Repository File Structure

```
TARA-FL/
├── aggregation/                # Aggregation strategies & adaptive router
│   ├── adaptive_aggregator.py  # Risk-based strategy dispatch
│   ├── base_aggregator.py      # Abstract aggregator base class
│   └── strategies.py           # FedAvg, Robust Trimmed, Median implementations
├── attacks/                    # Adversarial poisoning implementations
│   ├── gradient_poison.py      # Gradient scale and sign flip attacks
│   └── label_flip.py           # Dataset wrapper for label flipping
├── clients/                    # Client local training & parameter management
│   └── client.py               # Local SGD trainer & differential update extractor
├── data/                       # Dataset loading & partitioning
│   └── dataset.py              # MNIST loader, IID & Non-IID Dirichlet splits
├── demo/                       # Interactive browser dashboard & demo
│   ├── index.html              # Single-file app: Real Results (live + published) & Simulation tabs
│   ├── results/                # Published real experiment results (GitHub Pages, Mode B)
│   └── README.md               # How to run & what to try
├── detection/                  # Anomaly & outlier detectors
│   ├── anomaly_detector.py     # Base detector interface
│   ├── multimetric_detector.py # Multi-metric distance & angle detector
│   └── pid_detector.py         # Proportional-Integral-Derivative detector
├── experiments/                # Experiment runners & evaluation matrices
│   ├── analyze_matrix_results.py # Statistical synthesis & plotting script
│   ├── experiment_config.py    # Declarative configuration dataclass
│   ├── results_export.py       # Real-results JSON exporter (live dashboard feed)
│   ├── final_matrix_summary.csv # Master evaluation matrix CSV
│   ├── final_matrix_summary.json# Master evaluation matrix JSON
│   ├── matrix_raw/             # 20 raw round-by-round trajectory CSVs
│   ├── matrix_results/         # 20 scenario summary JSON files
│   ├── rigorous_analysis_summary.csv # Comprehensive multi-metric summary
│   ├── rigorous_analysis_summary.json# Full statistical analysis JSON
│   ├── repeated_comparison.py  # Multi-seed repeated runner
│   ├── run_baseline_fedavg.py  # Standalone FedAvg baseline runner
│   ├── run_experiment.py       # Standalone single experiment runner
│   ├── run_final_matrix.py     # 20-scenario master matrix runner
│   └── run_unified.py          # Unified 4-method comparative CLI runner (+ live export)
├── model/                      # Neural network architectures
│   ├── base_model.py           # Base model interface
│   └── model.py                # MNIST CNN and CIFAR-10 ConvNet
├── plots/                      # Generated visualization plots
│   ├── analysis/               # 8 Publication-grade evaluation figures
│   └── matrix/                 # Byzantine degradation & resilience charts
├── reports/                    # Generated Markdown & HTML reports
│   ├── EXPERIMENT_REPORT.md    # Single experiment Markdown report
│   ├── FINAL_MATRIX_REPORT.md  # 20-scenario master matrix Markdown report
│   └── report.html             # Interactive HTML report
├── risk/                       # Risk assessment & threat classification
│   ├── round_risk.py           # Multi-signal composite risk calculator
│   └── threat_classifier.py    # Threat typology classifier
├── server/                     # Server orchestrator & coordinator
│   └── server.py               # Orchestrates training, trust, risk, & evaluation
├── tests/                      # Full unit & integration test suite (92 tests)
├── trustengine/                # Continuous dynamic trust & quarantine
│   ├── probe_evaluator.py      # Server probe dataset evaluator
│   ├── quarantine.py           # QuarantineManager & TrustZone state machine
│   ├── trust_engine.py         # Dynamic trust calculation engine
│   └── trust_history.py        # Temporal trust score & anomaly history
├── visualization/              # Visualization, dashboard, & reporting
│   ├── dashboard.py            # Built-in HTTP REST API & web dashboard (CSV-based)
│   ├── live_server.py          # FastAPI + SSE backend streaming real experiment rounds
│   ├── publish_results.py      # Publisher: dashboard_data -> demo/results (static hosting)
│   ├── plots.py                # Plotting utilities for trajectories & bars
│   └── report_generator.py     # HTML & Markdown report synthesizer
├── main.py                     # Primary pipeline entry point
├── requirements.txt            # Dependency specification
└── README.md                   # Project documentation & reference
```

---

## 9. Empirical Summary & Key Findings

From the 160-run evaluation across 20 experimental scenarios:
- **Gradient Scaling ($\times 10$)**: FedAvg catastrophically fails ($22.65\%$ at 30% Byzantine); TARA-FL maintains **$90.00\%$** (IID) and **$83.33\%$** (Non-IID) by evicting malicious clients via QuarantineManager.
- **Sign Flipping ($-1.0\times$)**: FedAvg drops to $62.00\%$ in Non-IID; TARA-FL maintains **$81.97\%$** ($+19.98\%$ gain) via angular anomaly detection and Median/Robust routing.
- **Clean Baseline Trade-off**: Negligible impact in IID ($-0.27\%$); small utility penalty in Non-IID ($-3.23\%$) due to natural client heterogeneity triggering mild robust trimming.
- **Computational Overhead**: Server trust calculation adds $<0.15\text{s}$ per round, demonstrating negligible scaling cost over standard SGD training.
