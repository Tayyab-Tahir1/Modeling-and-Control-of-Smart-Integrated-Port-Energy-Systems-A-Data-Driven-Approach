# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from wandb import Api
from tqdm import tqdm

from MPC_model import (
    mpc_step,
    compute_max_power,
    BATTERY_CAPACITY,
    H2_STORAGE_CAPACITY,
    ENERGY_PER_KG_H2,
    HORIZON
)

# ========== CONFIG ==========
PROJECT    = "MPC_Imitation_LSTM_Sweep"
DATA_PATH  = "dataset.csv"
CACHE_FILE = "expert_sequences.pkl"
# ============================

def get_expert_action(flows):
    threshold = 1e-3
    vals = {
        1: flows.get('battery_to_load', 0),
        2: flows.get('pv_to_battery', 0),
        3: flows.get('hydrogen_produced', 0) * ENERGY_PER_KG_H2,
        4: flows.get('h2_to_load', 0),
        5: flows.get('grid_to_battery', 0),
        6: flows.get('h2_to_battery', 0),
        7: flows.get('h2_to_load_purchased', 0)
    }
    act, mx = 0, 0.0
    for a, v in vals.items():
        if v > mx and v > threshold:
            mx, act = v, a
    return act

def build_finetuning_sequences(df: pd.DataFrame, seq_len: int):
    max_power     = compute_max_power(df)
    max_tou       = df['Tou_Tariff'].max()
    max_fit       = df['FiT'].max()
    max_h2_tariff = df['H2_Tariff'].max()
    N = len(df)
    seqs = []
    for i in tqdm(range(N - seq_len - HORIZON + 1), desc="Building sequences"):
        soc, h2 = BATTERY_CAPACITY * 0.5, 0.0
        states, actions = [], []
        valid = True

        for t in range(i, i + seq_len):
            row = df.iloc[t]
            state = np.array([
                row['Load'] / max_power,
                row['PV'] / max_power,
                row['Tou_Tariff'] / max_tou,
                row['FiT'] / max_fit,
                row['H2_Tariff'] / max_h2_tariff,
                soc / BATTERY_CAPACITY,
                h2 / H2_STORAGE_CAPACITY,
                row['Day'] / 6.0,
                row['Hour'] / 23.0
            ], dtype=np.float32)

            if t + HORIZON > N:
                valid = False
                break

            forecast = df.iloc[t : t + HORIZON]
            out = mpc_step(forecast, soc, h2)
            act = 0
            if out:
                act = get_expert_action(out.get('flows', {}))
                soc = out.get('SoC', soc)
                h2  = out.get('H2_Storage', h2)

            states.append(state)
            actions.append(act)

        if valid and len(states) == seq_len:
            seqs.append((np.stack(states), np.array(actions, dtype=np.int64)))
    return seqs

class ImitationLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, num_actions=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

def train_lstm(model, sequences, epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # No scheduler patience � we train full epochs
    # If you still want LR decay, you can add a StepLR or similar here.

    X = torch.tensor([s for s, a in sequences], dtype=torch.float32).to(device)
    Y = torch.tensor([a for s, a in sequences], dtype=torch.long).to(device)
    N = X.shape[0]

    best_loss = float('inf')
    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            logits = model(X[idx])
            loss   = criterion(logits.view(-1, logits.shape[-1]), Y[idx].view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / ((N + batch_size - 1) // batch_size)
        best_loss = min(best_loss, avg_loss)
        wandb.log({
            "epoch": epoch + 1,
            "average_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    return model, best_loss

def sweep_train():
    run = wandb.init()
    cfg = run.config

    df = pd.read_csv(DATA_PATH)
    df.rename(columns={'Tou Tariff': 'Tou_Tariff', 'H2 Tariff': 'H2_Tariff'}, inplace=True)

    # Always load or build once into the same CACHE_FILE
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            sequences = pickle.load(f)
    else:
        sequences = build_finetuning_sequences(df, cfg.seq_len)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(sequences, f)

    model = ImitationLSTM(input_dim=9, hidden_dim=cfg.hidden_dim, num_actions=8)
    model, best_loss = train_lstm(
        model,
        sequences,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr
    )

    wandb.log({"train/best_loss": best_loss})
    fname = f"lstm_h{cfg.hidden_dim}_s{cfg.seq_len}.pt"
    torch.save(model.state_dict(), fname)
    wandb.save(fname)
    run.finish()

# ---------------------------------
# Grid sweep config (no patience)
# ---------------------------------
sweep_config = {
    "method": "grid",
    "metric": {"name": "train/best_loss", "goal": "minimize"},
    "parameters": {
        "seq_len":    {"values": [12, 24, 48]},
        "hidden_dim": {"values": [32, 64, 128]},
        "batch_size": {"values": [16, 32, 64, 128]},
        "lr":         {"values": [
            1e-5, 5e-5, 1e-4, 5e-4, 1e-3,
            5e-3, 1e-2, 5e-2, 1e-1, 5e-1
        ]},
        "epochs":     {"values": [10, 20, 50, 100]}
    }
}

if __name__ == "__main__":
    # 1) Create the grid sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT)

    # 2) Dispatch one job per grid point
    wandb.agent(sweep_id, function=sweep_train)

    # 3) Final analysis job for parallel-coordinates heat-map
    analysis = wandb.init(project=PROJECT, job_type="analysis")
    api      = Api()
    entity   = analysis.entity
    runs     = api.runs(f"{entity}/{PROJECT}", {"sweep": sweep_id})

    cols = ["seq_len", "hidden_dim", "batch_size", "lr", "epochs", "best_loss"]
    data = []
    for r in runs:
        c = r.config
        data.append([
            c["seq_len"],
            c["hidden_dim"],
            c["batch_size"],
            c["lr"],
            c["epochs"],
            r.summary.get("train/best_loss")
        ])

    table = wandb.Table(data=data, columns=cols)
    pc = wandb.plot.parallel_coordinates(
        table,
        dimensions=cols[:-1],
        config={"color": "best_loss", "colormap": "viridis"}
    )
    wandb.log({"sweep_parallel_heatmap": pc})
    analysis.finish()
