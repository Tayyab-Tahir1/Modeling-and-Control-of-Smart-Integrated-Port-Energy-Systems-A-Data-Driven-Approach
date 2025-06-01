import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import wandb

from models.environment import EnergyEnv

# 1) Hyperparameter options (first 4 only)
pv_options = ["PVoption1", "PVoption2", "PVoption3", "PVoption4"]
battery_options = [2750.0, 5500.0, 8250.0, 11000.0]       # in kWh
h2_options = [2000.0, 4000.0, 6000.0, 8000.0]             # in kg

# 2) Training regimen
episodes_per_model = 50
DATA_PATH = "dataset.csv"

# 3) Output directory
OUT_DIR = "Trained models"
os.makedirs(OUT_DIR, exist_ok=True)

# 4) Simple Gym wrapper passing sizing through to EnergyEnv
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
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        s, r, done, info = self.env.step(action)
        return np.array(s, dtype=np.float32), r, done, False, info

    def render(self, mode="human"):
        soc = self.env.soc / self.env.battery_capacity * 100
        print(f"Step {self.env.current_step} — SoC: {soc:.2f}%")

# 5) TensorBoard callback (optional)
class TBCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

# 6) Learning rate schedule
def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return schedule

# 7) Initialize W&B
data = pd.read_csv(DATA_PATH)
steps_per_episode = len(data)
total_models = len(pv_options) * len(battery_options) * len(h2_options)

wandb.init(
    project="Thesis_sizing_sweep",
    name="resource_size_sweep",
    config={
        "total_models": total_models,
        "episodes_per_model": episodes_per_model,
        "steps_per_episode": steps_per_episode
    }
)

# 8) Loop through all combinations
model_count = 0
for pv_col in pv_options:
    for batt_cap in battery_options:
        for h2_cap in h2_options:
            model_count += 1
            # Log progress & hyperparameters
            wandb.log({"models_done": model_count - 1}, commit=False)
            wandb.log({
                "pv_column": pv_col,
                "battery_capacity": batt_cap,
                "h2_storage_capacity": h2_cap
            }, commit=True)

            # Prepare data & environment
            df = data.rename(columns={"Tou Tariff": "Tou_Tariff", "H2 Tariff": "H2_Tariff"})
            df["PV"] = df[pv_col]
            env = EnergyEnvGym(df, pv_col="PV", battery_capacity=batt_cap, h2_storage_capacity=h2_cap)
            check_env(env, warn=True)

            # Create DQN model
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=linear_schedule(1e-4, 1e-5),
                buffer_size=100_000,
                exploration_fraction=0.8,
                exploration_final_eps=0.1,
                gamma=0.99,
                batch_size=64,
                policy_kwargs={"net_arch": [256, 256]},
                verbose=0
            )

            # Train for fixed episodes
            for ep in range(episodes_per_model):
                model.learn(
                    total_timesteps=steps_per_episode,
                    reset_num_timesteps=(ep == 0)
                )
                wandb.log({
                    "episode_in_model": ep + 1,
                    "models_done": model_count - 1,
                    "total_models": total_models
                }, commit=True)

            # Save the trained model
            filename = f"dqn_pv{pv_col[-1]}_bat{int(batt_cap)}_h2{int(h2_cap)}"
            model.save(os.path.join(OUT_DIR, filename))
            wandb.log({"models_done": model_count}, commit=True)

# 9) Finish W&B run
print(f"Trained {total_models} models, saved in '{OUT_DIR}'.")
wandb.finish()
