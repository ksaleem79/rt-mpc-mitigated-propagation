RT-MPC Mitigated Propagation

This repository contains the implementation, experiments, and reproducibility materials accompanying the manuscript:

Uncertainty-Aware Real-Time MPC for Mitigating Adversarial Manipulated-Content Propagation in Cyber–Information Networks

Overview

This work proposes a real-time Model Predictive Control (RT-MPC) framework for mitigating manipulated-content propagation under:

adversarial disturbances,
uncertain detector confidence,
stochastic operating conditions,
bounded intervention actions.

The repository contains the final implementation used to generate the figures, tables, and quantitative results reported in the revised manuscript.

DOI and Reproducibility Archive

The versioned reproducibility archive is available via Zenodo:

https://doi.org/10.5281/zenodo.20108632

Repository Structure
rt-mpc-mitigated-propagation/
│
├── src/
│   ├── mpc_controller.py
│   ├── simulation_environment.py
│   └── ...
│
├── experiments/
│   ├── stress_test.py
│   ├── cumulative_experiment.py
│   ├── disturbance_correlation.py
│   ├── phase_portrait.py
│   ├── intervention_signals.py
│   ├── multirun_analysis.py
│   ├── runtime_experiment.py
│   ├── uncertainty_experiment.py
│   ├── quantitative_metrics.py
│   └── ablation_analysis.py
│
├── figures/
│   └── manuscript figures
│
├── results/
│   ├── csv outputs
│   ├── numpy outputs
│   └── quantitative metrics
│
├── requirements.txt
├── LICENSE
└── README.md

Installation

Create a Python environment and install dependencies:

pip install -r requirements.txt

Running Experiments
Figure 1 – Exposure Trajectories
python experiments/exposure_trajectories.py

Figure 2 – Uncertainty Robustness
python experiments/uncertainty_experiment.py

Figure 3 – Computation Time Evaluation
python experiments/runtime_experiment.py

Figure 4 – Adversarial Stress Test
python experiments/stress_test.py

Figure 5 – Cumulative Exposure
python experiments/cumulative_experiment.py

Figure 6 – Disturbance Response Correlation
python experiments/disturbance_correlation.py

Figure 7 – Phase Portrait
python experiments/phase_portrait.py

Figure 8 – Intervention Signals
python experiments/intervention_signals.py

Figure 9 – Multi-Run Robustness
python experiments/multirun_analysis.py

Quantitative Metrics
python experiments/quantitative_metrics.py

Ablation Analysis
python experiments/ablation_analysis.py

Reproducibility

The reported experiments:

use fixed random seeds,
generate deterministic outputs under the documented configuration,
reproduce the figures reported in the manuscript,
use the same MPC configuration described in the paper.

Generated outputs are stored in:

results/


including:

CSV result files,
NumPy arrays,
quantitative metrics,
experiment outputs used for manuscript figures.
Experimental Configuration

Unless otherwise stated:

Network size:           n = 30
Intervention channels:  m = 3
Prediction horizon:     N = 5
Confidence:             σ = 0.8
Control bounds:         [-6.0, 6.0]
Terminal weight:        wt = 1.0
Solver:                 SciPy SLSQP

Computational Environment

The reported experiments were generated using:

Apple Mac Studio
Apple Silicon architecture
32 GB RAM
macOS
Python 3.9
NumPy
SciPy
Matplotlib

Notes on Repository Organization

Following reviewer feedback during manuscript revision, experiment scripts were reorganized into the experiments/ directory to improve reproducibility and maintain consistency between the repository structure and manuscript documentation.

Affiliation: Center of Excellence in Information Assurance (CoEIA), King Saud University, Riyadh, Saudi Arabia.

License: MIT License

Citation: If you use this code or reproducibility package, please cite the associated manuscript and Zenodo archive.
