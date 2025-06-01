import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import wandb

from models.environment import EnergyEnv

class EnergyEnvGym(gym.Env):
    """Gymnasium wrapper around EnergyEnv that accepts cost/emission weights"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, cost_weight, emission_weight):
        super().__init__()
        self.env = EnergyEnv(data, cost_weight=cost_weight, emission_weight=emission_weight)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.array(obs, dtype=np.float32), reward, done, False, info

    def render(self, mode="human"):
        soc_pct = self.env.soc / self.env.battery_capacity * 100
        print(f"Step {self.env.current_step} — SoC: {soc_pct:.2f}%")

def linear_schedule(initial_value, final_value):
    """Returns a function for linear lr decay from initial to final."""
    def schedule(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return schedule

if __name__ == "__main__":
    # --- Load & preprocess data ---
    DATA_PATH = "dataset.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    data.rename(columns={"Tou Tariff": "Tou_Tariff", "H2 Tariff": "H2_Tariff"}, inplace=True)

    # --- Sweep configuration ---
    cost_weights = np.arange(0.0, 1.01, 0.1)
    emission_weights = np.arange(0.0, 1.01, 0.1)
    combos = [(round(c,1), round(e,1)) for c in cost_weights for e in emission_weights]

    total_models = len(combos)
    episodes_per_model = 50
    steps_per_episode = len(data)

    # --- Output directory ---
    OUT_DIR = "trained models"
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Initialize Weights & Biases ---
    wandb.init(
        project="Thesis_weight_sweep",
        name="cost_emission_weight_sweep",
        config={
            "total_models": total_models,
            "episodes_per_model": episodes_per_model,
            "steps_per_episode": steps_per_episode
        }
    )

    # --- Sweep & train ---
    for idx, (cw, ew) in enumerate(combos, start=1):
        # Log which combination we're training
        wandb.log({"models_done": idx-1}, commit=False)
        wandb.log({"cost_weight": cw, "emission_weight": ew}, commit=True)

        # Build and check environment
        env = EnergyEnvGym(data, cost_weight=cw, emission_weight=ew)
        check_env(env, warn=True)

        # Create DQN model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(1e-4, 1e-5),
            buffer_size=100000,
            exploration_fraction=0.8,
            exploration_final_eps=0.1,
            gamma=0.99,
            batch_size=64,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0
        )

        # Train over a fixed number of episodes
        for ep in range(episodes_per_model):
            model.learn(
                total_timesteps=steps_per_episode,
                reset_num_timesteps=(ep == 0)
            )
            # Log episode progress
            wandb.log({
                "episode_in_model": ep + 1,
                "models_done": idx - 1,
                "total_models": total_models
            })

        # Save the trained model
        filename = f"dqn_cw{cw}_ew{ew}.zip"
        model.save(os.path.join(OUT_DIR, filename))

        # Mark this model as done
        wandb.log({"models_done": idx}, commit=True)

    print(f"All {total_models} models trained and saved in '{OUT_DIR}'.")
    wandb.finish()
