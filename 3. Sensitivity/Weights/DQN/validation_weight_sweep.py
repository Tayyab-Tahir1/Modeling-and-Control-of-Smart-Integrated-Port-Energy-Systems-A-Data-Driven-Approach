# validation_weight_sweep.py

import os
import argparse
import numpy as np
import pandas as pd
import wandb
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

from models.environment import EnergyEnv

# --- Gym wrapper to pass weights into EnergyEnv ---
class EnergyEnvGym(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self, data, cost_weight, emission_weight):
        super().__init__()
        self.env = EnergyEnv(data, cost_weight=cost_weight, emission_weight=emission_weight)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.array(obs, dtype=np.float32), reward, done, False, info

def run_validation():
    run = wandb.init()
    cfg = run.config
    cw = cfg.cost_weight
    ew = cfg.emission_weight

    # load & preprocess dataset
    df = pd.read_csv("dataset.csv")
    df.rename(columns={"Tou Tariff": "Tou_Tariff", "H2 Tariff": "H2_Tariff"}, inplace=True)

    # prepare evaluation env
    eval_env = EnergyEnvGym(df, cost_weight=cw, emission_weight=ew)

    # locate and load the corresponding model
    model_file = f"trained models/dqn_cw{cw:.1f}_ew{ew:.1f}.zip"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")
    model = DQN.load(model_file, env=eval_env)

    # run one full episode to collect bill & emissions
    obs, _ = eval_env.reset()
    total_bill = 0.0
    total_em   = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = eval_env.step(action)
        total_bill += info["Bill"]
        total_em   += info["Emissions"]

    # average per timestep
    steps = eval_env.env.max_steps
    avg_bill     = total_bill / steps
    avg_emission = total_em   / steps

    # log metrics
    wandb.log({
        "cost_weight":     cw,
        "emission_weight": ew,
        "avg_bill":        avg_bill,
        "avg_emission":    avg_emission
    })

    # push to summary for parallel coordinates axes
    run.summary["avg_bill"]     = avg_bill
    run.summary["avg_emission"] = avg_emission

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DQN models over weight sweep")
    parser.add_argument(
        "--mode",
        choices=["cost", "emission", "joint"],
        default="joint",
        help="Which sweep: cost-only, emission-only, or joint over both weights"
    )
    args = parser.parse_args()

    # grid values 0.0 to 1.0 in 0.1 steps
    values = [round(x,1) for x in np.arange(0.0, 1.01, 0.1)]

    # determine total runs
    if args.mode in ("cost", "emission"):
        total = len(values)
    else:
        total = len(values) * len(values)

    # build sweep config
    sweep_config = {
        "project":   "Thesis_weight_sweep_DQN_validation",
        "method":    "grid",
        "parameters": {
            "total_models": {"value": total},
        }
    }

    if args.mode == "cost":
        sweep_config["name"] = "cost_validation_sweep"
        sweep_config["parameters"].update({
            "cost_weight":     {"values": values},
            "emission_weight": {"value": 1.0}
        })
    elif args.mode == "emission":
        sweep_config["name"] = "emission_validation_sweep"
        sweep_config["parameters"].update({
            "cost_weight":     {"value": 1.0},
            "emission_weight": {"values": values}
        })
    else:
        sweep_config["name"] = "joint_validation_sweep"
        sweep_config["parameters"].update({
            "cost_weight":     {"values": values},
            "emission_weight": {"values": values}
        })

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=run_validation)
