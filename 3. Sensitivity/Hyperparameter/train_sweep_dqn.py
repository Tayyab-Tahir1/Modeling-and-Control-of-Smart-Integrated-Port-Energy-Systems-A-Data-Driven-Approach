# -*- coding: utf-8 -*-

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import wandb
from wandb import Api
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from models.environment import EnergyEnv

# ========== CONFIG ==========
PROJECT    = "Thesis_DQN_Sweep_"
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

    # load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    # env
    env = EnergyEnvGym(df)
    check_env(env, warn=True)

    # model
    policy_kwargs = dict(net_arch=list(cfg.net_arch))
    model = DQN(
        "MlpPolicy", env,
        learning_rate=cfg.learning_rate,
        buffer_size=100_000,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs_sweep/"
    )

    # train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=TensorBoardLoggingCallback()
    )

    # save
    arch_size = cfg.net_arch[0]
    name = f"DQN_Sweep_{arch_size}"
    model.save(name)
    wandb.save(name + ".zip")

    # test
    rews = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        ep_r = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(a)
            ep_r += r
        rews.append(ep_r)

    mean_r = np.mean(rews)
    wandb.log({"test/mean_reward": mean_r})
    run.finish()

# ---------------------------------
# Grid sweep config
# ---------------------------------
sweep_config = {
    "method": "grid",
    "metric": {"name": "test/mean_reward", "goal": "maximize"},
    "parameters": {
        "batch_size":          {"values": [32, 64, 128]},
        "net_arch":            {"values": [[64], [128], [256]]},
        "learning_rate":       {"values": [1e-5, 5e-5, 1e-4, 1e-3]},
        "gamma":               {"values": [0.8, 0.9, 0.99]},
        "exploration_fraction":{"values": [0.5, 0.7, 0.9]},
        "exploration_final_eps":{"values": [0.01, 0.1, 0.2]},
    }
}

if __name__ == "__main__":
    # 1) create the grid sweep in your default entity
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=PROJECT
    )

    # 2) run one job per grid point (no count)
    wandb.agent(sweep_id, function=sweep_train)

    # 3) once all are done, do an analysis job to plot the heatmap
    analysis = wandb.init(project=PROJECT, job_type="analysis")
    api      = Api()

    # fetch runs for this sweep
    entity  = analysis.entity
    runs    = api.runs(f"{entity}/{PROJECT}", {"sweep": sweep_id})

    # build table
    cols = [
        "batch_size",
        "exploration_fraction",
        "exploration_final_eps",
        "gamma",
        "learning_rate",
        "net_arch",
        "mean_reward"
    ]
    data = []
    for r in runs:
        c = r.config
        data.append([
            c["batch_size"],
            c["exploration_fraction"],
            c["exploration_final_eps"],
            c["gamma"],
            c["learning_rate"],
            c["net_arch"][0],
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
