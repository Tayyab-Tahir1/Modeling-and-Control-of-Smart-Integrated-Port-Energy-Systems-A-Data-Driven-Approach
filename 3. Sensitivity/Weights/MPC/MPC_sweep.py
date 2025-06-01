# mpc_sweep.py

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import wandb

import MPC_model
from finetune_lstm_with_mpc import build_finetuning_sequences

# Make sure Experts folder exists
EXPERT_DIR = "Experts"
os.makedirs(EXPERT_DIR, exist_ok=True)

def run_mpc():
    # Start W&B run
    run = wandb.init()
    cfg = run.config

    # Hyperparams
    cw      = cfg.cost_weight
    ew      = cfg.emission_weight
    seq_len = cfg.seq_len
    total   = cfg.total_models

    # Override the MPC weights
    MPC_model.COST_WEIGHT     = cw
    MPC_model.EMISSION_WEIGHT = ew

    # Load & preprocess data
    df = pd.read_csv("dataset.csv")
    df.rename(columns={"Tou Tariff":"Tou_Tariff","H2 Tariff":"H2_Tariff"}, inplace=True)
    N = len(df)

    # Build & save expert sequences
    sequences = build_finetuning_sequences(df, seq_len=seq_len)
    expert_fname = f"expert_cw{cw}_ew{ew}.pkl"
    expert_path  = os.path.join(EXPERT_DIR, expert_fname)
    with open(expert_path, "wb") as f:
        pickle.dump(sequences, f)

    # Roll MPC to collect bills & emissions
    bills, emissions = [], []
    soc, h2 = MPC_model.BATTERY_CAPACITY * 0.5, 0.0
    for i in range(N):
        horizon = min(MPC_model.HORIZON, N - i)
        forecast = df.iloc[i:i+horizon].copy()
        out = MPC_model.mpc_step(forecast, soc, h2)

        if out is None:
            # fallback grid‑only
            load    = forecast.iloc[0]["Load"]
            pv      = forecast.iloc[0]["PV"]
            pv2load = min(pv, load)
            purchase = (load-pv2load)*forecast.iloc[0]["Tou_Tariff"]
            revenue  = max(0, pv-pv2load)*forecast.iloc[0]["FiT"]
            bill     = purchase - revenue
            ef0      = forecast.iloc[0].get("Emission_factor",
                        MPC_model.DEFAULT_EMISSION_FACTOR)/1000.0
            emission = (load-pv2load)*ef0
            soc_next, h2_next = soc, h2
        else:
            bill     = out["Bill"]
            emission = out["Emissions"]
            soc_next = out["SoC"]
            h2_next  = out["H2_Storage"]

        bills.append(bill)
        emissions.append(emission)
        soc, h2 = soc_next, h2_next

    avg_bill     = float(np.mean(bills))
    avg_emission = float(np.mean(emissions))

    # Count how many expert files are already saved (including this one)
    models_saved = len([f for f in os.listdir(EXPERT_DIR) if f.endswith('.pkl')])

    # Log everything
    wandb.log({
        "expert_path":   expert_path,
        "n_sequences":   len(sequences),
        "avg_bill":      avg_bill,
        "avg_emission":  avg_emission,
        "models_saved":  models_saved,
        "total_models":  total,
    })

    # Also push to summary for Parallel Coordinates
    run.summary["avg_bill"]     = avg_bill
    run.summary["avg_emission"] = avg_emission
    run.summary["models_saved"] = models_saved
    run.summary["total_models"] = total

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPC expert data sweep")
    parser.add_argument(
        "--mode",
        choices=["cost", "emission", "joint"],
        default="joint",
        help="Sweep type: cost-only, emission-only, or joint"
    )
    args = parser.parse_args()

    # grid values 0.0, 0.1, …, 1.0
    values = [round(x,1) for x in np.arange(0.0, 1.01, 0.1)]

    # determine total_models
    if args.mode == "cost" or args.mode == "emission":
        total = len(values)
    else:
        total = len(values) * len(values)

    # build sweep config
    sweep_config = {
        "project":   "Thesis_weight_sweep_MPC",
        "method":    "grid",
        "parameters": {
            "seq_len":      {"value": 24},
            "total_models": {"value": total}
        }
    }

    if args.mode == "cost":
        sweep_config["name"] = "cost_sweep"
        sweep_config["parameters"].update({
            "cost_weight":     {"values": values},
            "emission_weight": {"value": 1.0}
        })
    elif args.mode == "emission":
        sweep_config["name"] = "emission_sweep"
        sweep_config["parameters"].update({
            "cost_weight":     {"value": 1.0},
            "emission_weight": {"values": values}
        })
    else:
        sweep_config["name"] = "joint_sweep"
        sweep_config["parameters"].update({
            "cost_weight":     {"values": values},
            "emission_weight": {"values": values}
        })

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=run_mpc)
