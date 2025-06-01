# val_islanded_dqn.py

import os
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from stable_baselines3 import DQN

# --- System Parameters ---
# PV rating = 16.2 MW
BATTERY_CAPACITY     = 8250.0
C_RATE               = 1.0
SOC_MIN              = 0.2 * BATTERY_CAPACITY
SOC_MAX              = 0.8 * BATTERY_CAPACITY
EFFICIENCY           = 0.95

H2_STORAGE_CAPACITY  = 6000.0
ENERGY_PER_KG_H2     = 32.0
FUEL_CELL_EFFICIENCY = 0.5

# Toggle grid on/off
GRID_AVAILABLE = False  # False = true islanded (no grid flows), True = normal mode

# --- Helpers ---
def compute_max_power(df):
    return max(
        df['Load'].max(),
        df['PV'].max(),
        BATTERY_CAPACITY * C_RATE,
        2000  # H₂ power capacity reference
    )

def get_feasible_actions(load, pv, soc, h2_storage):
    actions = [0]
    if soc > SOC_MIN + 1e-5:
        actions.append(1)  # battery→load
    if pv > 0:
        actions.append(3)  # PV→H₂
    if soc < SOC_MAX - 1e-5:
        actions.append(2)  # PV→battery
        if GRID_AVAILABLE:
            actions.append(5)  # grid→battery
    if h2_storage > 0:
        actions.extend([4, 6])  # H₂→load, H₂→battery
    if GRID_AVAILABLE and load > 0:
        actions.append(7)  # purchase H₂ from grid
    return actions

# --- Main ---
if __name__ == "__main__":
    # Start W&B run
    run = wandb.init(project="Thesis_Islanded_DQN_val")

    # Load model & data
    model = DQN.load("dqn_energy_model_finetuned.zip")
    df    = pd.read_csv("dataset.csv")
    df.rename(columns={'Tou Tariff':'Tou_Tariff','H2 Tariff':'H2_Tariff'}, inplace=True)

    # Prepare result columns
    cols = [
        'pv_to_load','pv_to_battery','pv_to_grid',
        'battery_to_load','grid_to_load','grid_to_battery',
        'h2_to_load','hydrogen_produced','h2_to_battery',
        'h2_to_load_purchased','H2_Purchased_kg',
        'Purchase','Sell','Bill','Emissions','SoC','H2_Storage','Chosen_Action'
    ]
    for c in cols:
        df[c] = np.nan

    # Initial state
    soc      = BATTERY_CAPACITY * 0.5
    h2_store = 0.0
    max_p    = compute_max_power(df)

    # Iterate timesteps
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        load      = row['Load']
        pv        = row['PV']
        tou       = row['Tou_Tariff']
        fit       = row['FiT']
        h2_tariff = row['H2_Tariff']
        ef        = row.get('Emission_factor', 0.5) / 1000.0

        # Build state
        state = np.array([
            load/max_p,
            pv/max_p,
            tou/df['Tou_Tariff'].max(),
            fit/df['FiT'].max(),
            h2_tariff/df['H2_Tariff'].max(),
            soc/BATTERY_CAPACITY,
            h2_store/H2_STORAGE_CAPACITY,
            row['Day']/6.0,
            row['Hour']/23.0
        ], dtype=np.float32)

        # Choose action
        action, _ = model.predict(state, deterministic=True)
        if action not in get_feasible_actions(load, pv, soc, h2_store):
            action = np.random.choice(get_feasible_actions(load, pv, soc, h2_store))

        # Initialize all flows to zero
        pv_to_load = pv_to_battery = pv_to_grid = 0.0
        batt_to_load = grid_to_load = grid_to_batt = 0.0
        h2_to_load = hydrogen_produced = h2_to_battery = h2_to_load_purchased = H2_Purchased_kg = 0.0
        purchase = sell = bill = emission = 0.0

        # 1) PV → load
        pv_to_load = min(pv, load)
        rem_load   = load - pv_to_load
        rem_pv     = pv - pv_to_load

        # 2) Action effects
        if action == 1:
            # Battery discharge → load
            avail = (soc - SOC_MIN) * EFFICIENCY
            d = min(rem_load, avail)
            batt_to_load = d
            soc -= d / EFFICIENCY
            rem_load -= d

        elif action == 2:
            # PV → battery
            pv_to_battery = rem_pv
            soc += pv_to_battery * EFFICIENCY
            rem_pv = 0.0

        elif action == 3:
            # PV → H₂ production
            hydrogen_produced = rem_pv / ENERGY_PER_KG_H2
            h2_store = min(h2_store + hydrogen_produced, H2_STORAGE_CAPACITY)
            rem_pv = 0.0

        elif action == 4:
            # Stored H₂ → load
            avail_h2 = h2_store * ENERGY_PER_KG_H2
            d = min(rem_load, avail_h2)
            h2_to_load = d
            used_kg = d / ENERGY_PER_KG_H2
            h2_store -= used_kg
            rem_load -= d

        elif action == 5 and GRID_AVAILABLE:
            # Grid → battery (only if grid available)
            cap_rem = SOC_MAX - soc
            e = cap_rem * 0.5
            grid_to_batt = e / EFFICIENCY
            soc += e

        elif action == 6:
            # Stored H₂ → battery
            avail_h2 = h2_store * ENERGY_PER_KG_H2
            cap_rem  = SOC_MAX - soc
            conv     = min(avail_h2, cap_rem)
            h2_to_battery = conv * FUEL_CELL_EFFICIENCY
            soc += h2_to_battery
            h2_store -= conv / ENERGY_PER_KG_H2

        elif action == 7 and GRID_AVAILABLE:
            # Purchase H₂ to load (if grid available)
            H2_Purchased_kg = rem_load / (ENERGY_PER_KG_H2 * FUEL_CELL_EFFICIENCY)
            h2_to_load_purchased = rem_load
            purchase = H2_Purchased_kg * h2_tariff
            rem_load = 0.0

        # 3) In islanded mode, any rem_load → H₂
        if not GRID_AVAILABLE and rem_load > 0:
            # Use stored H₂ first
            use_h2 = min(h2_store * ENERGY_PER_KG_H2, rem_load)
            h2_to_load += use_h2
            h2_store -= use_h2 / ENERGY_PER_KG_H2
            rem_load -= use_h2

            # Then purchase H₂ for any leftover
            if rem_load > 1e-6:
                H2_req = rem_load / (ENERGY_PER_KG_H2 * FUEL_CELL_EFFICIENCY)
                h2_to_load_purchased += rem_load
                H2_Purchased_kg = H2_req
                purchase = H2_req * h2_tariff
                rem_load = 0.0

        # 4) No PV export or grid import/charge in islanded
        if GRID_AVAILABLE:
            pv_to_grid   = rem_pv
            grid_to_load = rem_load
        else:
            pv_to_grid = grid_to_load = grid_to_batt = 0.0
            sell = 0.0

        # 5) Financials & emissions
        if GRID_AVAILABLE:
            grid_cost = (grid_to_load + grid_to_batt) * tou
            sell      = pv_to_grid * fit
            bill      = grid_cost - sell + purchase
            emission  = (grid_to_load + grid_to_batt) * ef
        else:
            bill     = purchase
            emission = 0.0

        # Record into DataFrame
        df.at[idx,'pv_to_load']              = pv_to_load
        df.at[idx,'pv_to_battery']           = pv_to_battery
        df.at[idx,'pv_to_grid']              = pv_to_grid
        df.at[idx,'battery_to_load']         = batt_to_load
        df.at[idx,'grid_to_load']            = grid_to_load
        df.at[idx,'grid_to_battery']         = grid_to_batt
        df.at[idx,'h2_to_load']              = h2_to_load
        df.at[idx,'hydrogen_produced']       = hydrogen_produced
        df.at[idx,'h2_to_battery']           = h2_to_battery
        df.at[idx,'h2_to_load_purchased']    = h2_to_load_purchased
        df.at[idx,'H2_Purchased_kg']         = H2_Purchased_kg
        df.at[idx,'Purchase']                = purchase
        df.at[idx,'Sell']                    = sell
        df.at[idx,'Bill']                    = bill
        df.at[idx,'Emissions']               = emission
        df.at[idx,'SoC']                     = (soc / BATTERY_CAPACITY) * 100.0
        df.at[idx,'H2_Storage']              = (h2_store / H2_STORAGE_CAPACITY) * 100.0
        df.at[idx,'Chosen_Action']           = action

        # Log line plots to W&B
        wandb.log({
            'pv_to_load': pv_to_load,
            'pv_to_battery': pv_to_battery,
            'battery_to_load': batt_to_load,
            'h2_to_load': h2_to_load,
            'hydrogen_produced': hydrogen_produced,
            'h2_to_battery': h2_to_battery,
            'h2_to_load_purchased': h2_to_load_purchased,
            'H2_Purchased_kg': H2_Purchased_kg,
            'Purchase': purchase,
            'Sell': sell,
            'Bill': bill,
            'Emissions': emission,
            'SoC': (soc / BATTERY_CAPACITY) * 100.0,
            'H2_Storage': (h2_store / H2_STORAGE_CAPACITY) * 100.0
        }, step=idx)

    # Save CSV and end run
    out_csv = "islanded_validation_results.csv"
    df.to_csv(out_csv, index=False)
    wandb.save(out_csv)
    run.finish()
