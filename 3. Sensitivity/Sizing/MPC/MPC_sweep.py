# val_sizing_sweep_mpc.py

import os
import numpy as np
import pandas as pd
import wandb

import MPC_model
from finetune_lstm_with_mpc import build_finetuning_sequences  # for HORIZON and DEFAULT_EMISSION_FACTOR

def run_validation():
    run = wandb.init()
    cfg = run.config

    # Map PV index to capacity in MW
    pv_rating_map = {1: 5.4, 2: 10.8, 3: 16.2, 4: 21.6}
    pv_capacity_mw = pv_rating_map[cfg.pv_index]
    batt_cap_kwh   = cfg.battery_capacity
    h2_cap_kg      = cfg.h2_storage_capacity

    # Override MPC sizing constants
    MPC_model.BATTERY_CAPACITY    = batt_cap_kwh
    MPC_model.H2_STORAGE_CAPACITY = h2_cap_kg

    # Load & preprocess
    df = pd.read_csv("dataset.csv")
    df.rename(columns={"Tou Tariff":"Tou_Tariff","H2 Tariff":"H2_Tariff"}, inplace=True)
    df["PV"] = df[f"PVoption{cfg.pv_index}"]
    N = len(df)

    # Initialize state
    soc, h2 = batt_cap_kwh * 0.5, 0.0

    # Keys to accumulate
    alloc_keys = [
        "pv_to_load","pv_to_battery","pv_to_grid",
        "battery_to_load","grid_to_load","grid_to_battery",
        "h2_to_load","hydrogen_produced","h2_to_battery",
        "h2_to_load_purchased","H2_Purchased_kg"
    ]
    extra_keys = ["Purchase","Sell","Bill","Emissions","SoC","H2_Storage"]

    # Initialize sums
    sums = {k: 0.0 for k in alloc_keys + extra_keys}
    steps = 0

    # Roll out MPC
    for i in range(N):
        row = df.iloc[i]
        horizon = min(MPC_model.HORIZON, N - i)
        forecast = df.iloc[i:i+horizon].copy()
        out = MPC_model.mpc_step(forecast, soc, h2)

        if out is None:
            # fallback grid-only
            load = row["Load"]; pv = row["PV"]
            pv2load = min(pv, load)
            load_rem = load - pv2load
            pv_rem   = pv - pv2load

            flows = {k: 0.0 for k in alloc_keys}
            flows["pv_to_load"] = pv2load
            flows["grid_to_load"] = load_rem
            flows["pv_to_grid"] = pv_rem

            purchase = load_rem * row["Tou_Tariff"]
            sell     = pv_rem   * row["FiT"]
            bill     = purchase - sell
            ef       = row.get("Emission_factor",
                        MPC_model.DEFAULT_EMISSION_FACTOR) / 1000.0
            emission = load_rem * ef

            soc_next, h2_next = soc, h2
        else:
            flows = out.get("flows", {})
            # compute purchase & sell from flows:
            purchase = (flows.get("grid_to_load",0)+flows.get("grid_to_battery",0)) * row["Tou_Tariff"]
            sell     = flows.get("pv_to_grid",0) * row["FiT"]
            bill     = out.get("Bill", purchase - sell)
            emission = out.get("Emissions", 0.0)
            soc_next = out.get("SoC", soc)
            h2_next  = out.get("H2_Storage", h2)

        # accumulate allocations
        for k in alloc_keys:
            sums[k] += flows.get(k, 0.0)
        # accumulate extras
        sums["Purchase"]    += purchase
        sums["Sell"]        += sell
        sums["Bill"]        += bill
        sums["Emissions"]   += emission
        sums["SoC"]         += soc_next
        sums["H2_Storage"]  += h2_next

        soc, h2 = soc_next, h2_next
        steps += 1

    # compute averages
    avgs = {f"avg_{k}": sums[k] / steps for k in sums}

    # log to W&B
    metrics = {
        "pv_capacity_MW":         pv_capacity_mw,
        "battery_capacity_kWh":   batt_cap_kwh,
        "h2_storage_capacity_kg": h2_cap_kg,
        **avgs
    }
    wandb.log(metrics)

    # summary for parallel coordinates
    run.summary["pv_capacity_MW"]         = pv_capacity_mw
    run.summary["battery_capacity_kWh"]   = batt_cap_kwh
    run.summary["h2_storage_capacity_kg"] = h2_cap_kg
    for name, val in avgs.items():
        run.summary[name] = val

    run.finish()

if __name__ == "__main__":
    sweep_config = {
        "project":   "Thesis_sizing_validation_MPC",
        "method":    "grid",
        "parameters": {
            "pv_index":            {"values": [1,2,3,4]},
            "battery_capacity":    {"values":[2750.0,5500.0,8250.0,11000.0]},
            "h2_storage_capacity": {"values":[2000.0,4000.0,6000.0,8000.0]}
        }
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=run_validation)
