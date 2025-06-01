# -*- coding: utf-8 -*-

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import wandb
from wandb import Api
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from models.environment import EnergyEnv

# ========== CONFIG ==========
PROJECT    = "Thesis_PPO_Sweep_"
DATA_PATH  = "dataset.csv"
# ============================

DATA_LINES      = 8760
EPISODES        = 4
TOTAL_TIMESTEPS = DATA_LINES * EPISODES

class EnergyEnvGym(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self, data):
        super().__init__()
        self.env = EnergyEnv(data)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
        self.action_space      = gym.spaces.Discrete(8)

    def reset(self, **kwargs):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        s, r, done, info = self.env.step(action)
        return np.array(s, dtype=np.float32), r, done, False, info

    def render(self, mode="human"):
        soc = self.env.soc / self.env.battery_capacity * 100
        print(f"Step {self.env.current_step}, SoC: {soc:.2f}%")

class TensorBoardLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self.logger.record("custom/reward", reward)
        loss = self.locals.get("loss", None)
        if loss is not None:
            self.logger.record("custom/loss", loss)
        return True

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        'Tou Tariff': 'Tou_Tariff',
        'H2 Tariff' : 'H2_Tariff'
    })

def sweep_train():
    run = wandb.init()
    cfg = run.config

    # load & preprocess
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    # env setup
    env = EnergyEnvGym(df)
    check_env(env, warn=True)

    # build PPO with hyperparams from cfg
    policy_kwargs = dict(net_arch=list(cfg.net_arch))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=15,              # keep fixed
        gamma=cfg.gamma,
        clip_range=cfg.clip_range,
        ent_coef=0.01,            # keep fixed
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs_ppo_sweep/"
    )

    # train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=TensorBoardLoggingCallback()
    )

    # save artifact
    arch_size = cfg.net_arch[0]
    name = f"PPO_Sweep_{arch_size}"
    model.save(name)
    wandb.save(name + ".zip")

    # quick test for mean reward
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        ep_r = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            ep_r += r
        rewards.append(ep_r)
    mean_reward = np.mean(rewards)
    wandb.log({"test/mean_reward": mean_reward})

    run.finish()

# ---------------------------------
# GRID sweep config
# ---------------------------------
sweep_config = {
    "method": "grid",
    "metric": {"name": "test/mean_reward", "goal": "maximize"},
    "parameters": {
        "batch_size":   {"values": [32, 64, 128]},
        "net_arch":     {"values": [[64], [128], [256]]},
        "learning_rate":{"values": [1e-5, 5e-5, 1e-4, 1e-3]},
        "gamma":        {"values": [0.8, 0.9, 0.99]},
        "n_steps":      {"values": [1024, 2048, 4096]},
        "clip_range":   {"values": [0.1, 0.2, 0.3]}
    }
}

if __name__ == "__main__":
    # 1) create the grid sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=PROJECT
    )

    # 2) launch one job per grid point (no count)
    wandb.agent(sweep_id, function=sweep_train)

    # 3) analysis job to log the heat-map parallel coordinates
    analysis = wandb.init(project=PROJECT, job_type="analysis")
    api      = Api()
    entity   = analysis.entity
    runs     = api.runs(f"{entity}/{PROJECT}", {"sweep": sweep_id})

    # collect all runs into a table
    cols = [
        "batch_size",
        "net_arch",
        "learning_rate",
        "gamma",
        "n_steps",
        "clip_range",
        "mean_reward"
    ]
    data = []
    for r in runs:
        c = r.config
        data.append([
            c["batch_size"],
            c["net_arch"][0],
            c["learning_rate"],
            c["gamma"],
            c["n_steps"],
            c["clip_range"],
            r.summary.get("test/mean_reward")
        ])

    table = wandb.Table(data=data, columns=cols)
    pc    = wandb.plot.parallel_coordinates(
        table,
        dimensions=cols[:-1],
        config={"color": "mean_reward", "colormap": "viridis"}
    )
    wandb.log({"sweep_parallel_heatmap": pc})
    analysis.finish()
