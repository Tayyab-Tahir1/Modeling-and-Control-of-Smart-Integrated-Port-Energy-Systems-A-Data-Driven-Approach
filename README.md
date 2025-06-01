# Modeling and Control of Smart Integrated Port Energy Systems: A Data‑Driven Approach
**EMJMD‑MIR Master's Thesis Companion Repository**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)  
![Stable‑Baselines3](https://img.shields.io/badge/RL-stable--baselines3-2.2.1-ff69b4)

---

## ✨ What’s inside?

This repository bundles **all code, data sets, trained models, simulations and post‑processing notebooks** that accompany my Erasmus Mundus Joint Master Degree (EMJMD) project on *data‑driven modelling and control of smart integrated port energy systems*.

Key artefacts:

* A custom **Gymnasium** environment that captures the multi‑vector dynamics of a port: shore‑power load, battery storage, hydrogen production, on‑site PV/wind generation and grid interaction.
* End‑to‑end training pipelines for **reinforcement‑learning** (DQN & PPO), **model predictive control (MPC)** baselines and **supervised learning surrogates**.
* Scenario studies covering peak‑shaving, vessel‑arrival uncertainty, demand‑response participation and different European electricity mixes.
* A techno‑economic post‑processor computing Levelised Cost of Energy (LCOE), payback, emission abatement and revenue streams.

---

## 🗂️ Repository layout

```
Datasets/                       → Raw & processed data (vessel calls, load, weather, tariffs, emissions)
  ├── Base_Case.csv
  ├── Scenario_*.csv            (peak, DR, RES‑share…) 
  └── Sensitivity/              (grid‑sweeps summary CSVs)

1. Environment/                → OpenAI Gymnasium port‑energy environment
   ├── port_env.py              (core env)
   └── components/              (battery, electrolyser, cranes…)

2. Controllers/                → Control algorithms
   ├── RL/                      (DQN, PPO training + checkpoints)
   ├── MPC/                     (Pyomo & CVXPY scripts)
   └── Surrogates/              (LSTM & XGBoost demand predictors)

3. Experiments/                → Reproducible experiment configs (Hydra)
   ├── Base_Case/
   ├── Peak_Shaving/
   └── Demand_Response/

4. Sensitivity/                → Hyper‑parameter & sizing sweeps (Weights & Biases)

5. Tecno_Economic/             → CAPEX / OPEX assumptions + notebooks
```

Each top‑level folder contains a local `README.md` with CLI examples and expected outputs.

---

## 🚀 Quick start

1. **Clone** and create the environment (Conda recommended):

```bash
git clone https://github.com/Tayyab-Tahir1/Modeling-and-Control-of-Smart-Integrated-Port-Energy-Systems-A-Data-Driven-Approach.git
cd smart‑port‑energy
conda env create -f environment.yml
conda activate smart‑port‑energy
```

2. **Run the base‑case RL controller**:

```bash
python Experiments/Base_Case/train_rl.py \
       --config Experiments/Base_Case/config.yaml \
       --project_name "MIR_Master_PortRL"
```

3. **Validate** a pre‑trained model:

```bash
python Controllers/RL/evaluate.py \
       --weights checkpoints/ppo_base_port.zip
```

4. **Launch a hyper‑parameter sweep** (Weights & Biases):

```bash
cd Sensitivity
wandb sweep sweep_rl.yaml
wandb agent <SWEEP‑ID>
```

---

## 🛠️ Main dependencies

| Package                | Tested version |
|------------------------|----------------|
| Python                 | 3.10.12        |
| gymnasium              | 0.29.1         |
| stable‑baselines3      | 2.2.1          |
| torch (+CUDA)          | 2.3.0          |
| wandb                  | 0.17.0         |
| pyomo / cvxpy          | 6.7.2 / 1.5.0  |
| pandas, numpy, matplotlib | latest LTS |

Create the full environment:

```yaml
# environment.yml
name: smart‑port‑energy
channels:
  - conda‑forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - gymnasium==0.29.1
      - stable‑baselines3==2.2.1
      - torch==2.3.0
      - wandb
      - pyomo
      - cvxpy
      - pandas
      - numpy
      - matplotlib
```

---

## 📊 Re‑creating the thesis figures

All experiments log their results to `Experiments/<name>/results/`.  
The helper scripts below regenerate Chapter‑level plots:

```bash
python utils/plot_training.py --logdir Experiments/Base_Case/results
python utils/plot_tecno_econ.py --input Tecno_Economic/output
```

---

## 🤝 Contributing

Bug fixes, new scenarios (e.g. cold ironing for cruise ships) and additional control strategies are welcome.  
Please open an issue first; ensure `pre‑commit` passes (`black`, `ruff`, `isort`).

---

## 📄 License & citation

Released under the permissive **MIT License**—see [LICENSE](./LICENSE).

If you build on this work, please cite:

```bibtex
@mastersthesis{Tahir2025,
  title   = {Modeling and Control of Smart Integrated Port Energy Systems: A Data‑Driven Approach},
  author  = {Tayyab Tahir},
  school  = {NTNU, EMJMD – Maritime Intelligent & Renewable (MIR) Programme},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

* Supervisor: Prof. Mehdi Zadeh (NTNU, Marine Technology)  
* Funding: Erasmus+, MIR consortium
* Kudos to the OpenAI Gymnasium, Stable‑Baselines3 and W&B communities.

Smooth sailing ⚓⚡
