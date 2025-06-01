# val_islanded_mpc_safe.py

import os
import pickle
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import cvxpy as cp

from finetune_lstm_with_mpc import build_finetuning_sequences

# --- System Parameters ---
BATTERY_CAPACITY     = 8250.0
C_RATE               = 1.0
SOC_MIN              = 0.2 * BATTERY_CAPACITY
SOC_MAX              = 0.8 * BATTERY_CAPACITY
EFFICIENCY           = 0.95

H2_STORAGE_CAPACITY  = 6000.0
ENERGY_PER_KG_H2     = 32.0
FUEL_CELL_EFFICIENCY = 0.5

DEFAULT_EMISSION_FACTOR = 0.5  # g/kWh

GRID_CHARGE_FRACTION = 0.5
COST_WEIGHT         = 1.0
EMISSION_WEIGHT     = 0.2
HORIZON             = 24

# Toggle islanded mode
GRID_AVAILABLE = False  # False=islanded, True=grid-connected

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def compute_max_power(df):
    return max(df['Load'].max(), df['PV'].max(), BATTERY_CAPACITY * C_RATE)

def safe_wandb_log(run, metrics: dict, step: int, state: dict):
    """
    Try to log to wandb twice, then disable logging if it still fails.
    state = {'enabled': bool}
    """
    if not state['enabled']:
        return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        print(f"W&B log failure at step {step}: {e}\nRetrying once...")
        try:
            run.log(metrics, step=step)
        except Exception as e2:
            print(f"Retry also failed: {e2}\nDisabling further W&B logging.")
            state['enabled'] = False

def safe_wandb_save(run, path: str, state: dict):
    if not state['enabled']:
        return
    try:
        run.save(path)
    except Exception as e:
        print(f"W&B save failure ({path}): {e}\nDisabling further W&B logging.")
        state['enabled'] = False

# -----------------------------------------------------------------------------
# MPC step (unchanged)
# -----------------------------------------------------------------------------
def mpc_step(forecast, soc0, h20):
    T = len(forecast)
    L   = forecast['Load'].to_numpy()
    PV  = forecast['PV'].to_numpy()
    Tou = forecast['Tou_Tariff'].to_numpy()
    FiT = forecast['FiT'].to_numpy()
    H2r = forecast['H2_Tariff'].to_numpy()
    if 'Emission_factor' in forecast:
        Em = forecast['Emission_factor'].to_numpy() / 1000.0
    else:
        Em = np.full(T, DEFAULT_EMISSION_FACTOR / 1000.0)

    pv2load = np.minimum(PV, L)
    surplus = PV - pv2load

    # decision vars
    grid_load   = cp.Variable(T, nonneg=True)
    grid_charge = cp.Variable(T, nonneg=True)
    batt_dis    = cp.Variable(T, nonneg=True)
    pv_charge   = cp.Variable(T, nonneg=True)
    pv2h2       = cp.Variable(T, nonneg=True)
    h2_use      = cp.Variable(T, nonneg=True)
    h2_to_batt  = cp.Variable(T, nonneg=True)
    h2_pur      = cp.Variable(T, nonneg=True)

    soc = cp.Variable(T+1)
    h2  = cp.Variable(T+1)

    cons = [soc[0] == soc0, h2[0] == h20]
    for t in range(T):
        cons += [
            pv_charge[t] + pv2h2[t] <= surplus[t],
            pv2load[t] + batt_dis[t] + h2_use[t] + h2_pur[t] + grid_load[t] == L[t],
            soc[t+1] == soc[t]
                        - batt_dis[t]/EFFICIENCY
                        + (pv_charge[t]+grid_charge[t])*EFFICIENCY
                        + h2_to_batt[t],
            soc[t+1] >= SOC_MIN, soc[t+1] <= SOC_MAX,
            h2[t+1] == h2[t]
                        + pv2h2[t]/ENERGY_PER_KG_H2
                        - h2_use[t]/ENERGY_PER_KG_H2
                        - (h2_to_batt[t]/FUEL_CELL_EFFICIENCY)/ENERGY_PER_KG_H2,
            h2[t+1] >= 0, h2[t+1] <= H2_STORAGE_CAPACITY
        ]

    expo = [surplus[t] - (pv_charge[t] + pv2h2[t]) for t in range(T)]
    grid_cost = cp.sum(cp.multiply(grid_load+grid_charge, Tou))
    h2_cost   = cp.sum(cp.multiply(h2_pur/(ENERGY_PER_KG_H2*FUEL_CELL_EFFICIENCY), H2r))
    pv_rev    = cp.sum(cp.multiply(cp.hstack(expo), FiT))
    emis      = cp.sum(cp.multiply(grid_load+grid_charge, Em))

    obj = cp.Minimize(COST_WEIGHT*(grid_cost + h2_cost - pv_rev)
                     + EMISSION_WEIGHT*emis)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ["optimal","optimal_inaccurate"]:
        return None

    f = {
        'pv_to_load':        float(pv2load[0]),
        'pv_to_battery':     float(pv_charge.value[0]),
        'pv_to_grid':        float(expo[0]) if GRID_AVAILABLE else 0.0,
        'battery_to_load':   float(batt_dis.value[0]),
        'grid_to_load':      float(grid_load.value[0]) if GRID_AVAILABLE else 0.0,
        'grid_to_battery':   float(grid_charge.value[0]) if GRID_AVAILABLE else 0.0,
        'h2_to_load':        float(h2_use.value[0]),
        'hydrogen_produced': float(pv2h2.value[0] / ENERGY_PER_KG_H2),
        'h2_to_battery':     float(h2_to_batt.value[0]),
        'h2_to_load_purchased': float(h2_pur.value[0]),
        'H2_Purchased_kg':   float(h2_pur.value[0] / (ENERGY_PER_KG_H2*FUEL_CELL_EFFICIENCY))
    }

    grid_c = (f['grid_to_load']+f['grid_to_battery'])*Tou[0] if GRID_AVAILABLE else 0.0
    pv_r   = expo[0]*FiT[0] if GRID_AVAILABLE else 0.0
    h2_c   = f['H2_Purchased_kg']*H2r[0]
    bill   = grid_c - pv_r + h2_c
    emis0  = (f['grid_to_load']+f['grid_to_battery'])*Em[0] if GRID_AVAILABLE else 0.0

    return {
        'flows': f,
        'Purchase': grid_c + h2_c,
        'Sell': pv_r,
        'Bill': bill,
        'Emissions': emis0,
        'SoC': float(soc.value[1]),
        'H2_Storage': float(h2.value[1]),
        'Chosen_Action': 'MPC'
    }

# -----------------------------------------------------------------------------
# Main validation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # init wandb
    run = wandb.init(project="Thesis_Islanded_MPC_val")
    log_state = {'enabled': True}

    # load & preprocess
    df = pd.read_csv("dataset.csv")
    df.rename(columns={'Tou Tariff':'Tou_Tariff','H2 Tariff':'H2_Tariff'}, inplace=True)

    # save expert sequences (always)
    seqs = build_finetuning_sequences(df, seq_len=HORIZON)
    with open("expert_sequences.pkl","wb") as f:
        pickle.dump(seqs, f)
    safe_wandb_save(run, "expert_sequences.pkl", log_state)

    # prepare output frame
    cols = [
      'pv_to_load','pv_to_battery','pv_to_grid',
      'battery_to_load','grid_to_load','grid_to_battery',
      'h2_to_load','hydrogen_produced','h2_to_battery',
      'h2_to_load_purchased','H2_Purchased_kg',
      'Purchase','Sell','Bill','Emissions','SoC','H2_Storage','Chosen_Action'
    ]
    for c in cols:
        df[c] = np.nan

    soc = BATTERY_CAPACITY * 0.5
    h2  = 0.0

    # iterate
    for i, row in tqdm(df.iterrows(), total=len(df)):
        out = mpc_step(df.iloc[i:i+HORIZON].copy(), soc, h2)
        if out is None:
            # fallback: islanded-only H2
            L, P = row['Load'], row['PV']
            pv2l = min(P,L); rem = L-pv2l
            h2l = min(h2*ENERGY_PER_KG_H2, rem)
            h2 -= h2l/ENERGY_PER_KG_H2; rem -= h2l
            h2p = rem/(ENERGY_PER_KG_H2*FUEL_CELL_EFFICIENCY) if rem>0 else 0.0
            cost = h2p*row['H2_Tariff']
            f = {
              'pv_to_load':pv2l,'pv_to_battery':0,'pv_to_grid':0,
              'battery_to_load':0,'grid_to_load':0,'grid_to_battery':0,
              'h2_to_load':h2l,'hydrogen_produced':0,
              'h2_to_battery':0,
              'h2_to_load_purchased':rem,'H2_Purchased_kg':h2p
            }
            out = {
              'flows':f,'Purchase':cost,'Sell':0.0,
              'Bill':cost,'Emissions':0.0,
              'SoC':soc,'H2_Storage':h2,'Chosen_Action':'Fallback'
            }

        # record
        for k,v in out['flows'].items():
            df.at[i,k] = v
        df.at[i,'Purchase']     = out['Purchase']
        df.at[i,'Sell']         = out['Sell']
        df.at[i,'Bill']         = out['Bill']
        df.at[i,'Emissions']    = out['Emissions']
        df.at[i,'SoC']          = (out['SoC']/BATTERY_CAPACITY)*100
        df.at[i,'H2_Storage']   = (out['H2_Storage']/H2_STORAGE_CAPACITY)*100
        df.at[i,'Chosen_Action']= out['Chosen_Action']

        # safe logging
        metrics = {**out['flows'],
                   'Purchase':out['Purchase'],
                   'Sell':out['Sell'],
                   'Bill':out['Bill'],
                   'Emissions':out['Emissions'],
                   'SoC':(out['SoC']/BATTERY_CAPACITY)*100,
                   'H2_Storage':(out['H2_Storage']/H2_STORAGE_CAPACITY)*100}
        safe_wandb_log(run, metrics, step=i, state=log_state)

        soc = out['SoC']
        h2  = out['H2_Storage']

    # save results CSV regardless
    csv_path = "islanded_mpc_validation.csv"
    df.to_csv(csv_path, index=False)
    safe_wandb_save(run, csv_path, log_state)

    run.finish()
