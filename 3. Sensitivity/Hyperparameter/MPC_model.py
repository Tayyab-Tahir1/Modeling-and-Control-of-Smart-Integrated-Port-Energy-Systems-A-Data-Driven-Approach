import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm

# =========================
# System Parameters (must match your EnergyEnv parameters)
# =========================
BATTERY_CAPACITY = 5500       # kWh
C_RATE = 0.5
SOC_MIN = 0.2 * BATTERY_CAPACITY
SOC_MAX = 0.8 * BATTERY_CAPACITY
EFFICIENCY = 0.95

H2_STORAGE_CAPACITY = 3000.0  # kg (adjust as needed)
ENERGY_PER_KG_H2 = 32.0       # kWh per kg H2
FUEL_CELL_EFFICIENCY = 0.5    # 50% conversion efficiency
H2_POWER_CAPACITY = 2000      # kW (used for normalization)

# Default emission factor (if not provided in data): in gram/kWh.
# It will be converted to kg/kWh.
DEFAULT_EMISSION_FACTOR = 0.5

GRID_CHARGE_FRACTION = 0.5

# Cost and emission weights (tunable)
COST_WEIGHT = 1.0
EMISSION_WEIGHT = 0.2

# MPC horizon (number of time steps for the lookahead)
HORIZON = 24

# =========================
# Compute Normalization Reference
# =========================
def compute_max_power(df):
    # Use the maximum among Load, PV, battery power (BATTERY_CAPACITY * C_RATE), and H2 power capacity.
    return max(df['Load'].max(), df['PV'].max(), BATTERY_CAPACITY * C_RATE, H2_POWER_CAPACITY)

# =========================
# MPC Optimization Function
# =========================
def mpc_step(forecast_df, soc_init, h2_init):
    """
    Solves an MPC optimization over a horizon T using cvxpy.
    
    forecast_df: DataFrame with forecast data for T steps (must contain:
                 'Load', 'PV', 'Tou_Tariff', 'FiT', 'H2_Tariff', and optionally 'Emission_factor').
    soc_init: initial battery state-of-charge (kWh)
    h2_init: initial hydrogen storage (kg)
    
    Returns a dictionary containing:
      - flows (first-step decisions)
      - Purchase, Sell, Bill, Emissions for t=0,
      - updated battery SoC (SoC) and hydrogen storage (H2_Storage),
      - a label 'Chosen_Action' set to "MPC".
    """
    T = forecast_df.shape[0]
    
    # Extract forecast data as numpy arrays.
    Load = forecast_df['Load'].to_numpy()       # kWh
    PV = forecast_df['PV'].to_numpy()             # kWh available from PV
    Tou = forecast_df['Tou_Tariff'].to_numpy()      # cost per kWh (grid import)
    FiT = forecast_df['FiT'].to_numpy()            # revenue per kWh (PV export)
    H2_rate = forecast_df['H2_Tariff'].to_numpy()    # cost per kg hydrogen

    # Get emission factor array: if provided in forecast data, convert from gram/kWh to kg/kWh;
    # otherwise, use the default.
    if "Emission_factor" in forecast_df.columns:
        Emission = forecast_df["Emission_factor"].to_numpy() / 1000.0
    else:
        Emission = np.full(T, DEFAULT_EMISSION_FACTOR / 1000.0)

    # Assume PV first meets load.
    pv_to_load = np.minimum(PV, Load)
    PV_surplus = PV - pv_to_load  # available for battery charging or hydrogen production

    # Decision variables for t = 0,..., T-1.
    grid_load = cp.Variable(T, nonneg=True)      # kWh from grid to meet load
    grid_charge = cp.Variable(T, nonneg=True)      # kWh from grid used for battery charging
    batt_discharge = cp.Variable(T, nonneg=True)   # kWh discharged from battery to supply load
    pv_charge = cp.Variable(T, nonneg=True)        # kWh from PV surplus used for battery charging
    pv_to_H2 = cp.Variable(T, nonneg=True)         # kWh from PV surplus used for hydrogen production
    h2_usage = cp.Variable(T, nonneg=True)         # kWh from stored hydrogen used directly to meet load
    h2_to_batt = cp.Variable(T, nonneg=True)         # kWh from converting stored hydrogen to battery energy
    H2_purchase = cp.Variable(T, nonneg=True)        # kWh of load met by purchased hydrogen (after conversion)

    # State variables: battery SoC (kWh) and hydrogen storage (kg)
    soc = cp.Variable(T+1)
    h2 = cp.Variable(T+1)
    
    constraints = []
    constraints += [soc[0] == soc_init]
    constraints += [h2[0] == h2_init]
    
    for t in range(T):
        # PV surplus allocation constraint.
        constraints += [pv_charge[t] + pv_to_H2[t] <= PV_surplus[t]]
        # Load balance: load must be met by PV-to-load, battery discharge, hydrogen usage, purchased H2, or grid load.
        constraints += [pv_to_load[t] + batt_discharge[t] + h2_usage[t] + H2_purchase[t] + grid_load[t] == Load[t]]
        # Battery dynamics:
        constraints += [soc[t+1] == soc[t] - batt_discharge[t] / EFFICIENCY + (pv_charge[t] + grid_charge[t]) * EFFICIENCY + h2_to_batt[t]]
        constraints += [soc[t+1] >= SOC_MIN, soc[t+1] <= SOC_MAX]
        # Hydrogen dynamics:
        constraints += [h2[t+1] == h2[t] + (pv_to_H2[t] / ENERGY_PER_KG_H2)
                                    - (h2_usage[t] / ENERGY_PER_KG_H2)
                                    - ((h2_to_batt[t] / FUEL_CELL_EFFICIENCY) / ENERGY_PER_KG_H2)]
        constraints += [h2[t+1] >= 0, h2[t+1] <= H2_STORAGE_CAPACITY]
    
    # Compute PV export (unused surplus) for revenue.
    PV_export = cp.hstack([PV_surplus[t] - (pv_charge[t] + pv_to_H2[t]) for t in range(T)])
    
    # Cost components per time step.
    grid_cost = cp.multiply((grid_load + grid_charge), Tou)
    H2_purchase_cost = cp.multiply(H2_purchase / (ENERGY_PER_KG_H2 * FUEL_CELL_EFFICIENCY), H2_rate)
    PV_revenue = cp.multiply(PV_export, FiT)
    emissions = cp.multiply((grid_load + grid_charge), Emission)
    
    composite = cp.sum(cp.multiply(COST_WEIGHT, grid_cost + H2_purchase_cost - PV_revenue) +
                       cp.multiply(EMISSION_WEIGHT, emissions))
    
    objective = cp.Minimize(composite)
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print("ECOS solver failed:", e)
        print("Trying SCS solver...")
        prob.solve(solver=cp.SCS, verbose=False)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: MPC optimization did not find an optimal solution.")
        return None

    # Extract first-step decisions.
    flows = {
        'pv_to_load': pv_to_load[0],
        'pv_to_battery': pv_charge.value[0],
        'pv_to_grid': PV_export.value[0],
        'battery_to_load': batt_discharge.value[0],
        'grid_to_load': grid_load.value[0],
        'grid_to_battery': grid_charge.value[0],
        'h2_to_load': h2_usage.value[0],
        'hydrogen_produced': pv_to_H2.value[0] / ENERGY_PER_KG_H2,
        'h2_to_battery': h2_to_batt.value[0],
        'h2_to_load_purchased': H2_purchase.value[0],
        'H2_Purchased_kg': H2_purchase.value[0] / (ENERGY_PER_KG_H2 * FUEL_CELL_EFFICIENCY)
    }
    grid_cost0 = (grid_load.value[0] + grid_charge.value[0]) * Tou[0]
    pv_revenue0 = PV_export.value[0] * FiT[0]
    H2_purchase_cost0 = (H2_purchase.value[0] / (ENERGY_PER_KG_H2 * FUEL_CELL_EFFICIENCY)) * H2_rate[0]
    bill = grid_cost0 - pv_revenue0 + H2_purchase_cost0
    # For emissions at t=0, use the first forecast row's emission factor if available.
    if "Emission_factor" in forecast_df.columns:
        emission_factor0 = forecast_df.iloc[0]["Emission_factor"] / 1000.0
    else:
        emission_factor0 = DEFAULT_EMISSION_FACTOR / 1000.0
    emissions0 = (grid_load.value[0] + grid_charge.value[0]) * emission_factor0
    
    soc_next = soc.value[1]
    h2_next = h2.value[1]
    
    out = {
        'flows': flows,
        'Purchase': grid_cost0,
        'Sell': pv_revenue0,
        'Bill': bill,
        'Emissions': emissions0,
        'SoC': soc_next,
        'H2_Storage': h2_next,
        'Chosen_Action': "MPC"
    }
    return out

# =========================
# Main MPC Validation Script
# =========================
def main():
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    # Rename columns to match the expected names.
    rename_mapping = {'Tou Tariff': 'Tou_Tariff', 'H2 Tariff': 'H2_Tariff'}
    df.rename(columns=rename_mapping, inplace=True)
    print("Dataset loaded successfully.")
    
    global self_max_power
    self_max_power = compute_max_power(df)
    
    results = []
    soc = BATTERY_CAPACITY * 0.5  # initial battery SoC
    h2_storage = 0.0              # initial hydrogen storage
    
    N = df.shape[0]
    horizon = HORIZON
    
    # Sequentially roll forward over the dataset.
    for i in tqdm(range(N)):
        T = min(horizon, N - i)
        forecast = df.iloc[i:i+T].copy()
        
        mpc_out = mpc_step(forecast, soc, h2_storage)
        if mpc_out is None:
            # If MPC fails, use a fallback (grid-only supply)
            flows = {
                'pv_to_load': min(forecast.iloc[0]['PV'], forecast.iloc[0]['Load']),
                'pv_to_battery': 0,
                'pv_to_grid': max(0, forecast.iloc[0]['PV'] - min(forecast.iloc[0]['PV'], forecast.iloc[0]['Load'])),
                'battery_to_load': 0,
                'grid_to_load': forecast.iloc[0]['Load'] - min(forecast.iloc[0]['PV'], forecast.iloc[0]['Load']),
                'grid_to_battery': 0,
                'h2_to_load': 0,
                'hydrogen_produced': 0,
                'h2_to_battery': 0,
                'h2_to_load_purchased': 0,
                'H2_Purchased_kg': 0
            }
            Purchase = (forecast.iloc[0]['Load'] - min(forecast.iloc[0]['PV'], forecast.iloc[0]['Load'])) * forecast.iloc[0]['Tou_Tariff']
            Sell = max(0, forecast.iloc[0]['PV'] - min(forecast.iloc[0]['PV'], forecast.iloc[0]['Load'])) * forecast.iloc[0]['FiT']
            Bill = Purchase - Sell
            if "Emission_factor" in forecast.columns:
                emission_factor0 = forecast.iloc[0]["Emission_factor"] / 1000.0
            else:
                emission_factor0 = DEFAULT_EMISSION_FACTOR / 1000.0
            Emissions = (forecast.iloc[0]['Load'] - min(forecast.iloc[0]['PV'], forecast.iloc[0]['Load'])) * emission_factor0
            soc_next = soc
            h2_next = h2_storage
            chosen_action = "Grid Only"
        else:
            flows = mpc_out['flows']
            Purchase = mpc_out['Purchase']
            Sell = mpc_out['Sell']
            Bill = mpc_out['Bill']
            Emissions = mpc_out['Emissions']
            soc_next = mpc_out['SoC']
            h2_next = mpc_out['H2_Storage']
            chosen_action = mpc_out['Chosen_Action']
        
        # Merge original row with MPC outputs.
        res_dict = df.iloc[i].to_dict()
        res_dict.update({
            'pv_to_load': flows['pv_to_load'],
            'pv_to_battery': flows['pv_to_battery'],
            'pv_to_grid': flows['pv_to_grid'],
            'battery_to_load': flows['battery_to_load'],
            'grid_to_load': flows['grid_to_load'],
            'grid_to_battery': flows['grid_to_battery'],
            'h2_to_load': flows['h2_to_load'],
            'hydrogen_produced': flows['hydrogen_produced'],
            'h2_to_battery': flows['h2_to_battery'],
            'h2_to_load_purchased': flows['h2_to_load_purchased'],
            'H2_Purchased_kg': flows['H2_Purchased_kg'],
            'Purchase': Purchase,
            'Sell': Sell,
            'Bill': Bill,
            'Emissions': Emissions,
            'SoC': (soc_next / BATTERY_CAPACITY) * 100,
            'H2_Storage': (h2_next / H2_STORAGE_CAPACITY) * 100,
            'Chosen_Action': chosen_action
        })
        results.append(res_dict)
        soc = soc_next
        h2_storage = h2_next

    results_df = pd.DataFrame(results)
    output_csv = "results_mpc.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Plot key metrics.
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(results_df["Purchase"].rolling(24).mean(), label="Purchase")
    plt.plot(results_df["Sell"].rolling(24).mean(), label="Sell")
    plt.plot(results_df["Bill"].rolling(24).mean(), label="Net Bill")
    plt.plot(results_df["Emissions"].rolling(24).mean(), label="Emissions")
    plt.title("24-hour Rolling Average of Financial Metrics (MPC)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(results_df["SoC"], label="Battery SoC")
    plt.plot(results_df["H2_Storage"], label="Hydrogen Storage (%)")
    plt.axhline(y=20, color="r", linestyle="--", label="Min SoC")
    plt.axhline(y=80, color="r", linestyle="--", label="Max SoC")
    plt.title("Battery SoC & Hydrogen Storage (MPC)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("energy_management_results_mpc.png")
    plt.close()
    print("Plots saved to 'energy_management_results_mpc.png'.")

if __name__ == "__main__":
    main()
