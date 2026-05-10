# RT-MPC Mitigated Propagation

This repository contains the implementation and experiments for the paper:

**“Real-Time MPC-Based Mitigation of Manipulated Content Propagation Under Uncertainty”**

---

## 📌 Overview

This work proposes a Model Predictive Control (RT-MPC) framework to mitigate the spread of manipulated content under:

- adversarial disturbances
- uncertain detection confidence
- stochastic system behavior

---

## 📁 Repository Structure

src/           # Core MPC and simulation modules
experiments/   # Scripts for generating experimental results
figures/       # Plot scripts
results/       # Saved simulation outputs

---

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt

---

### 2. Run experiments

**Figure 5 (Stress Test):**

python -m experiments.stress_test

**Figure 6 (Cumulative Exposure):**

python -m experiments.cumulative_experiment

**Figure 7 (Disturbance Correlation):**

python -m experiments.disturbance_correlation

**Figure 8 (Phase Portrait):**

python -m experiments.phase_portrait

**Figure 9 (Intervention Signals):**

python -m experiments.intervention_signals

**Figure 10 (Multi-run Analysis):**

python -m experiments.multirun_analysis

---

## 📊 Reproducibility

All figures in the paper can be reproduced using the scripts in the `experiments/` folder.

Generated output is stored in:

results/

---

## 📜 License

CoEIA-KSU License


## 🏛️ Affiliation

## License
This project is licensed under the MIT License.

## Affiliation
Developed at the Center of Excellence in Information Assurance (CoEIA), King Saud University.
