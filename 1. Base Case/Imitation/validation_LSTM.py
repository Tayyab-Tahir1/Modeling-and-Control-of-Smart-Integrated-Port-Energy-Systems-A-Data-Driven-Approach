# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

# =========================
# System Parameters (consistent with MPC and training)
# =========================
BATTERY_CAPACITY = 5500       # kWh (updated from 5000 to 5500 to match training)
C_RATE = 0.5
SOC_MIN = 0.2 * BATTERY_CAPACITY
SOC_MAX = 0.8 * BATTERY_CAPACITY
EFFICIENCY = 0.95

# Hydrogen storage parameters
H2_STORAGE_CAPACITY = 3000.0  # Maximum hydrogen storage in kg
ENERGY_PER_KG_H2 = 32.0       # kWh per kg H2
FUEL_CELL_EFFICIENCY = 0.5    # 50% fuel cell efficiency
H2_POWER_CAPACITY = 2000      # Fuel cell power capacity

# Default emission factor (if not provided in data): in gram/kWh.
EMISSION_FACTOR = 0.5

# Grid battery charging fraction
GRID_CHARGE_FRACTION = 0.5

# Global reference for maximum power (set later)
self_max_power = None

def compute_max_power(df):
    # Use maximum among Load, PV, battery power (BATTERY_CAPACITY * C_RATE), and H2 power capacity.
    return max(df['Load'].max(), df['PV'].max(), BATTERY_CAPACITY * C_RATE, H2_POWER_CAPACITY)

# =========================
# Feasible Actions Function
# =========================
def get_feasible_actions_new(load, pv, tou_tariff, h2_tariff, soc, h2_storage):
    """
    Expanded action space (0-7):
      0: Do nothing
      1: Battery discharge to supply load
      2: Charge battery using PV
      3: Produce hydrogen from PV
      4: Use stored hydrogen directly to meet load
      5: Charge battery using grid power
      6: Convert stored hydrogen to battery energy (H2-to-Battery)
      7: Purchase hydrogen from grid to meet load
    """
    feasible = [0]
    if soc > SOC_MIN + 1e-5:
        feasible.append(1)
    if soc < SOC_MAX - 1e-5:
        feasible.append(2)
        feasible.append(5)
    if pv > 0:
        feasible.append(3)
    if h2_storage > 0:
        feasible.append(4)
        feasible.append(6)
    if load > 0:
        feasible.append(7)
    return feasible

# =========================
# Process Action Function
# =========================
def process_action_new(action, load, pv, tou_tariff, fit, h2_tariff, soc, h2_storage, emission_factor_used):
    """
    Processes the chosen action and computes flows, cost, emissions, etc.
    Returns updated soc, h2_storage, allocations, grid_cost, pv_revenue, bill, emissions, and reward.
    """
    allocations = {
        'pv_to_load': 0.0,
        'pv_to_battery': 0.0,
        'pv_to_grid': 0.0,
        'battery_to_load': 0.0,
        'grid_to_load': 0.0,
        'grid_to_battery': 0.0,
        'h2_to_load': 0.0,
        'hydrogen_produced': 0.0,
        'h2_to_battery': 0.0,
        'h2_to_load_purchased': 0.0,
        'H2_Purchased_kg': 0.0
    }
    H2_purchase_cost = 0  # Cost incurred if hydrogen is purchased
    
    # 1) Use PV to supply load
    allocations['pv_to_load'] = min(pv, load)
    load_remaining = load - allocations['pv_to_load']
    pv_remaining = pv - allocations['pv_to_load']
    
    fc_eff = FUEL_CELL_EFFICIENCY  # 0.5
    
    if action == 0:
        pass
    elif action == 1:
        available_energy = (soc - SOC_MIN) * EFFICIENCY
        allocations['battery_to_load'] = min(load_remaining, available_energy)
        soc -= allocations['battery_to_load'] / EFFICIENCY
        load_remaining -= allocations['battery_to_load']
    elif action == 2:
        allocations['pv_to_battery'] = pv_remaining
        soc += allocations['pv_to_battery'] * EFFICIENCY
        pv_remaining = 0
    elif action == 3:
        energy_used = pv_remaining
        allocations['hydrogen_produced'] = energy_used / ENERGY_PER_KG_H2
        h2_storage = min(h2_storage + allocations['hydrogen_produced'], H2_STORAGE_CAPACITY)
        pv_remaining = 0
    elif action == 4:
        available_h2_energy = h2_storage * ENERGY_PER_KG_H2
        allocations['h2_to_load'] = min(load_remaining, available_h2_energy)
        hydrogen_used = allocations['h2_to_load'] / ENERGY_PER_KG_H2
        h2_storage -= hydrogen_used
        load_remaining -= allocations['h2_to_load']
    elif action == 5:
        available_capacity = SOC_MAX - soc
        energy_to_charge = available_capacity * GRID_CHARGE_FRACTION
        allocations['grid_to_battery'] = energy_to_charge / EFFICIENCY
        soc += energy_to_charge
    elif action == 6:
        available_h2_energy = h2_storage * ENERGY_PER_KG_H2
        battery_capacity_remaining = SOC_MAX - soc
        energy_converted = min(available_h2_energy, battery_capacity_remaining)
        battery_energy_gained = energy_converted * fc_eff
        allocations['h2_to_battery'] = battery_energy_gained
        soc += battery_energy_gained
        hydrogen_used = energy_converted / ENERGY_PER_KG_H2
        h2_storage -= hydrogen_used
    elif action == 7:
        hydrogen_required_kg = load_remaining / (ENERGY_PER_KG_H2 * fc_eff)
        allocations['H2_Purchased_kg'] = hydrogen_required_kg
        allocations['h2_to_load_purchased'] = load_remaining
        H2_purchase_cost = hydrogen_required_kg * h2_tariff
        load_remaining = 0

    allocations['grid_to_load'] = load_remaining
    allocations['pv_to_grid'] = pv_remaining
    
    # Clamp SoC within the limits.
    soc = max(SOC_MIN, min(soc, SOC_MAX))
    
    grid_cost = (allocations['grid_to_load'] + allocations['grid_to_battery']) * tou_tariff
    pv_revenue = allocations['pv_to_grid'] * fit
    bill = grid_cost - pv_revenue + H2_purchase_cost
    
    emissions = (allocations['grid_to_load'] + allocations['grid_to_battery']) * emission_factor_used
    
    composite = bill + emissions
    max_possible_bill = self_max_power * max(tou_tariff, h2_tariff)
    if max_possible_bill == 0:
        max_possible_bill = 1
    reward = - composite / max_possible_bill
    
    return soc, h2_storage, allocations, grid_cost, pv_revenue, bill, emissions, reward

# =========================
# LSTM Imitation Model Definition (same as training)
# =========================
class ImitationLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, num_actions=8):
        super(ImitationLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        returns: logits of shape (batch_size, seq_len, num_actions)
        """
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits

# =========================
# Main Validation Script Using the Trained LSTM Model
# =========================
def main():
    # Load the trained LSTM imitation model
    model_path = "lstm_mpc_imitation.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = ImitationLSTM(input_dim=9, hidden_dim=256, num_actions=8).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found: {model_path}")
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.eval()
    print(f"Loaded LSTM imitation model from {model_path}")
    
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
    
    # Rename columns to match expected names.
    df.rename(columns={'Tou Tariff': 'Tou_Tariff', 'H2 Tariff': 'H2_Tariff'}, inplace=True)
    
    global self_max_power
    self_max_power = compute_max_power(df)
    
    # Set up columns for outputs.
    allocation_columns = [
        'pv_to_load', 'pv_to_battery', 'pv_to_grid',
        'battery_to_load', 'grid_to_load', 'grid_to_battery',
        'h2_to_load', 'hydrogen_produced', 'h2_to_battery',
        'h2_to_load_purchased', 'H2_Purchased_kg',
        'Purchase', 'Sell', 'Bill', 'Emissions', 'SoC', 'H2_Storage', 'Chosen_Action'
    ]
    for col in allocation_columns:
        df[col] = np.nan

    soc = BATTERY_CAPACITY * 0.5
    h2_storage = 0.0
    
    # We'll use a rolling window to form the LSTM input.
    seq_len = 24  # for example, a 24-hour window
    state_history = []  # to store the sequence of states

    print("\nRunning validation with LSTM imitation model...")
    # Loop over each time step.
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        load = row['Load']
        pv = row['PV']
        tou_tariff = row['Tou_Tariff']
        fit = row['FiT']
        h2_tariff = row['H2_Tariff']
        day = row['Day']
        hour = row['Hour']
        
        # Determine emission factor (convert grams/kWh to kg/kWh)
        if 'Emission_factor' in row:
            emission_factor_used = row['Emission_factor'] / 1000.0
        else:
            emission_factor_used = EMISSION_FACTOR / 1000.0
        
        # Build current state vector (normalized)
        current_state = np.array([
            load / self_max_power,
            pv / self_max_power,
            tou_tariff / df['Tou_Tariff'].max(),
            fit / df['FiT'].max(),
            h2_tariff / df['H2_Tariff'].max(),
            soc / BATTERY_CAPACITY,
            h2_storage / H2_STORAGE_CAPACITY,
            day / 6.0,
            hour / 23.0
        ], dtype=np.float32)
        
        # Append current state to the history.
        state_history.append(current_state)
        # Ensure the history length is seq_len (pad with first state if needed)
        if len(state_history) < seq_len:
            padded_history = [state_history[0]] * (seq_len - len(state_history)) + state_history
        else:
            padded_history = state_history[-seq_len:]
        
        # Convert sequence to tensor of shape (1, seq_len, input_dim)
        seq_tensor = torch.tensor(np.array(padded_history), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = lstm_model(seq_tensor)  # shape: (1, seq_len, num_actions)
            # Take the logits corresponding to the last time step in the sequence.
            last_logits = logits[0, -1, :]
            predicted_action = int(torch.argmax(last_logits).item())
        
        feasible_actions = get_feasible_actions_new(load, pv, tou_tariff, h2_tariff, soc, h2_storage)
        # If predicted action is not feasible, choose one randomly.
        if predicted_action not in feasible_actions:
            predicted_action = int(np.random.choice(feasible_actions))
        
        # Process the chosen action.
        soc, h2_storage, allocations, purchase, sell, bill, emissions, reward = process_action_new(
            predicted_action, load, pv, tou_tariff, fit, h2_tariff, soc, h2_storage, emission_factor_used
        )
        
        # Record outputs in the dataframe.
        for key, value in allocations.items():
            df.at[index, key] = value
        df.at[index, 'Purchase'] = purchase
        df.at[index, 'Sell'] = sell
        df.at[index, 'Bill'] = bill
        df.at[index, 'Emissions'] = emissions
        df.at[index, 'SoC'] = (soc / BATTERY_CAPACITY) * 100
        df.at[index, 'H2_Storage'] = (h2_storage / H2_STORAGE_CAPACITY) * 100
        df.at[index, 'Chosen_Action'] = predicted_action

    output_csv = "results_hybrid_lstm.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Plot key metrics.
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df["Purchase"].rolling(24).mean(), label="Purchase")
    plt.plot(df["Sell"].rolling(24).mean(), label="Sell")
    plt.plot(df["Bill"].rolling(24).mean(), label="Net Bill")
    plt.plot(df["Emissions"].rolling(24).mean(), label="Emissions")
    plt.title("24-hour Rolling Average of Financial Metrics (LSTM Imitation)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df["SoC"], label="Battery SoC")
    plt.plot(df["H2_Storage"], label="Hydrogen Storage (%)")
    plt.axhline(y=20, color="r", linestyle="--", label="Min SoC")
    plt.axhline(y=80, color="r", linestyle="--", label="Max SoC")
    plt.title("Battery SoC & Hydrogen Storage")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("energy_management_results_hybrid_lstm.png")
    plt.close()
    print("Plots saved to 'energy_management_results_hybrid_lstm.png'.")

if __name__ == "__main__":
    main()
