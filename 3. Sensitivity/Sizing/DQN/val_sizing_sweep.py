# val_sizing_sweep.py

import os
import numpy as np
import pandas as pd
import wandb
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

from models.environment import EnergyEnv

# --- Gym wrapper passing sizing into EnergyEnv ---
class EnergyEnvGym(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self, data, pv_col, battery_capacity, h2_storage_capacity):
        super().__init__()
        self.env = EnergyEnv(
            data,
            pv_col=pv_col,
            battery_capacity=battery_capacity,
            h2_storage_capacity=h2_storage_capacity
        )
        self.observation_space = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

    def reset(self, **kwargs):
        state = self.env.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        # EnergyEnv.step returns: next_state, reward, done, info
        state, reward, done, info = self.env.step(action)
        return np.array(state, dtype=np.float32), reward, done, info

def run_validation():
    run = wandb.init()
    cfg = run.config

    # Map PV index to actual capacity (MW)
    pv_rating_map = {1: 5.4, 2: 10.8, 3: 16.2, 4: 21.6}
    pv_capacity_mw = pv_rating_map[cfg.pv_index]
    batt_cap_kwh  = cfg.battery_capacity
    h2_cap_kg     = cfg.h2_storage_capacity

    # load & preprocess data
    df = pd.read_csv("dataset.csv")
    df.rename(columns={"Tou Tariff": "Tou_Tariff", "H2 Tariff": "H2_Tariff"}, inplace=True)
    df["PV"] = df[f"PVoption{cfg.pv_index}"]

    # build env
    env = EnergyEnvGym(
        df,
        pv_col="PV",
        battery_capacity=batt_cap_kwh,
        h2_storage_capacity=h2_cap_kg
    )
    obs, _ = env.reset()

    # load model
    model_file = f"Trained models/dqn_pv{cfg.pv_index}_bat{int(batt_cap_kwh)}_h2{int(h2_cap_kg)}.zip"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")
    model = DQN.load(model_file, env=env)

    # keys to accumulate
    alloc_keys = [
        "pv_to_load", "pv_to_battery", "pv_to_grid",
        "battery_to_load", "grid_to_load", "grid_to_battery",
        "h2_to_load", "hydrogen_produced", "h2_to_battery",
        "h2_to_load_purchased", "H2_Purchased_kg"
    ]
    extra_keys = ["Purchase", "Sell", "Bill", "Emissions", "SoC"]

    # initialize sums
    sums = {k: 0.0 for k in alloc_keys + extra_keys}
    steps = 0
    done = False

    # run one full episode
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        for k in sums:
            sums[k] += info.get(k, 0.0)
        steps += 1

    # compute averages
    avgs = {f"avg_{k}": sums[k] / steps for k in sums}

    # log metrics
    metrics = {
        "pv_capacity_MW": pv_capacity_mw,
        "battery_capacity_kWh": batt_cap_kwh,
        "h2_storage_capacity_kg": h2_cap_kg,
        **avgs
    }
    wandb.log(metrics)

    # summary for parallel coordinates
    run.summary["pv_capacity_MW"] = pv_capacity_mw
    run.summary["battery_capacity_kWh"] = batt_cap_kwh
    run.summary["h2_storage_capacity_kg"] = h2_cap_kg
    for name, val in avgs.items():
        run.summary[name] = val

    run.finish()

if __name__ == "__main__":
    sweep_config = {
        "project": "Thesis_sizing_sweep_validation",
        "method": "grid",
        "parameters": {
            "pv_index":             {"values": [1, 2, 3, 4]},
            "battery_capacity":     {"values": [2750.0, 5500.0, 8250.0, 11000.0]},
            "h2_storage_capacity":  {"values": [2000.0, 4000.0, 6000.0, 8000.0]}
        }
    }

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=run_validation)
