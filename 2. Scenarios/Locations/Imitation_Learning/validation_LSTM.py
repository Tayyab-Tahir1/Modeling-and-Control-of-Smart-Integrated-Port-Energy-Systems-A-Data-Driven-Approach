import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb

# =========================
# System Parameters (updated)
# =========================
BATTERY_CAPACITY = 5000       # kWh
C_RATE = 0.5
SOC_MIN = 0.2 * BATTERY_CAPACITY
SOC_MAX = 0.8 * BATTERY_CAPACITY
EFFICIENCY = 0.95

# Hydrogen storage parameters
H2_STORAGE_CAPACITY = 3000.0  # Maximum hydrogen storage in kg
ENERGY_PER_KG_H2 = 32.0       # kWh per kg H2
FUEL_CELL_EFFICIENCY = 0.5    # Fuel cell conversion efficiency
H2_POWER_CAPACITY = 2000      # Fuel cell power capacity

# Default emission factor (if not provided in data): in gram/kWh.
DEFAULT_EMISSION_FACTOR = 0.5  

# Grid battery charging fraction
GRID_CHARGE_FRACTION = 0.5

# Global reference for maximum power (computed from dataset)
self_max_power = None

def compute_max_power(df):
    return max(df['Load'].max(), df['PV'].max(), BATTERY_CAPACITY * C_RATE, H2_POWER_CAPACITY)

# =========================
# Feasible Actions Function
# =========================
def get_feasible_actions_new(load, pv, tou_tariff, h2_tariff, soc, h2_storage):
    """
    Expanded action space (0–7):
      0: Do nothing
      1: Battery discharge to meet load
      2: Charge battery using excess PV
      3: Produce hydrogen using excess PV (store produced H2)
      4: Use stored hydrogen directly to meet load
      5: Charge battery using grid power
      6: Convert stored hydrogen to battery energy (H2-to-Battery)
      7: Purchase hydrogen (in kg) from the grid to meet load
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
def process_action_new(action, load, pv, tou_tariff, fit, h2_tariff, soc, h2_storage, emission_factor_used=DEFAULT_EMISSION_FACTOR/1000.0):
    """
    Processes the chosen action and computes flows, cost, emissions, etc.
    Returns:
      updated soc, updated h2_storage, allocations, grid_cost, pv_revenue, bill, emissions, reward.
    The allocations dictionary includes:
      'pv_to_load', 'pv_to_battery', 'pv_to_grid',
      'battery_to_load', 'grid_to_load', 'grid_to_battery',
      'h2_to_load', 'hydrogen_produced', 'h2_to_battery', 'h2_to_load_purchased',
      'H2_Purchased_kg'
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
    H2_purchase_cost = 0
    
    # Use PV to supply load.
    allocations['pv_to_load'] = min(pv, load)
    load_remaining = load - allocations['pv_to_load']
    pv_remaining = pv - allocations['pv_to_load']
    
    fc_eff = FUEL_CELL_EFFICIENCY  # typically 0.5
    
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
    
    soc = max(SOC_MIN, min(soc, SOC_MAX))
    
    grid_cost = (allocations['grid_to_load'] + allocations['grid_to_battery']) * tou_tariff
    pv_revenue = allocations['pv_to_grid'] * fit
    bill = grid_cost - pv_revenue + H2_purchase_cost
    
    # Calculate emissions using the provided emission factor (in kg/kWh)
    emissions = (allocations['grid_to_load'] + allocations['grid_to_battery']) * emission_factor_used
    composite = (1.0 * bill) + (0.2 * emissions)
    max_possible_bill = self_max_power * max(tou_tariff, h2_tariff)
    if max_possible_bill == 0:
        max_possible_bill = 1
    reward = - composite / max_possible_bill

    return soc, h2_storage, allocations, grid_cost, pv_revenue, bill, emissions, reward

# =========================
# LSTM Imitation Model Definition
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
# Modified Process Dataset Function for LSTM Imitation (with Merged Original Data)
# =========================
def process_dataset(country_code, dataset_path, lstm_model, device):
    # Read dataset and preserve all original columns.
    df_original = pd.read_csv(dataset_path)
    print(f"Dataset {dataset_path} loaded successfully for {country_code}.")
    
    # Rename columns to expected names.
    df_original.rename(columns={'Tou Tariff': 'Tou_Tariff', 'H2 Tariff': 'H2_Tariff'}, inplace=True)
    
    global self_max_power
    self_max_power = compute_max_power(df_original)
    
    # List of computed columns.
    computed_columns = [
        'pv_to_load', 'pv_to_battery', 'pv_to_grid',
        'battery_to_load', 'grid_to_load', 'grid_to_battery',
        'h2_to_load', 'hydrogen_produced', 'h2_to_battery',
        'h2_to_load_purchased', 'H2_Purchased_kg',
        'Purchase', 'Sell', 'Bill', 'Emissions', 'SoC', 'H2_Storage', 'Chosen_Action'
    ]
    
    results = []
    
    soc = BATTERY_CAPACITY * 0.5
    h2_storage = 0.0
    
    seq_len = 24  # 24-hour rolling window for LSTM input
    state_history = []
    
    print(f"\nRunning validation for {country_code} with the LSTM imitation model...")
    for index, row in tqdm(df_original.iterrows(), total=df_original.shape[0]):
        load = row['Load']
        pv = row['PV']
        tou_tariff = row['Tou_Tariff']
        fit = row['FiT']
        h2_tariff = row['H2_Tariff']
        day = row.get('Day', 0)
        hour = row.get('Hour', 0)
        
        if 'Emission_factor' in row:
            emission_factor_used = row['Emission_factor'] / 1000.0
        else:
            emission_factor_used = DEFAULT_EMISSION_FACTOR / 1000.0
        
        current_state = np.array([
            load / self_max_power,
            pv / self_max_power,
            tou_tariff / df_original['Tou_Tariff'].max(),
            fit / df_original['FiT'].max(),
            h2_tariff / df_original['H2_Tariff'].max(),
            soc / BATTERY_CAPACITY,
            h2_storage / H2_STORAGE_CAPACITY,
            day / 6.0,
            hour / 23.0
        ], dtype=np.float32)
        
        state_history.append(current_state)
        if len(state_history) < seq_len:
            padded_history = [state_history[0]] * (seq_len - len(state_history)) + state_history
        else:
            padded_history = state_history[-seq_len:]
        
        seq_tensor = torch.tensor(np.array(padded_history), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = lstm_model(seq_tensor)  # shape: (1, seq_len, num_actions)
            last_logits = logits[0, -1, :]
            predicted_action = int(torch.argmax(last_logits).item())
        
        feasible_actions = get_feasible_actions_new(load, pv, tou_tariff, h2_tariff, soc, h2_storage)
        if predicted_action not in feasible_actions:
            predicted_action = int(np.random.choice(feasible_actions))
        
        soc, h2_storage, allocations, purchase, sell, bill, emissions, reward = process_action_new(
            predicted_action, load, pv, tou_tariff, fit, h2_tariff, soc, h2_storage, emission_factor_used
        )
        
        # Merge original row with computed outputs.
        res_dict = row.to_dict()  # Preserves all original columns.
        res_dict.update({
            'pv_to_load': allocations['pv_to_load'],
            'pv_to_battery': allocations['pv_to_battery'],
            'pv_to_grid': allocations['pv_to_grid'],
            'battery_to_load': allocations['battery_to_load'],
            'grid_to_load': allocations['grid_to_load'],
            'grid_to_battery': allocations['grid_to_battery'],
            'h2_to_load': allocations['h2_to_load'],
            'hydrogen_produced': allocations['hydrogen_produced'],
            'h2_to_battery': allocations['h2_to_battery'],
            'h2_to_load_purchased': allocations['h2_to_load_purchased'],
            'H2_Purchased_kg': allocations['H2_Purchased_kg'],
            'Purchase': purchase,
            'Sell': sell,
            'Bill': bill,
            'Emissions': emissions,
            'SoC': (soc / BATTERY_CAPACITY) * 100,
            'H2_Storage': (h2_storage / H2_STORAGE_CAPACITY) * 100,
            'Chosen_Action': predicted_action
        })
        results.append(res_dict)
    
    df_out = pd.DataFrame(results)
    return df_out

# =========================
# Main Validation Script for Multiple Datasets (LSTM)
# =========================
def main():
    model_path = "lstm_mpc_imitation.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = ImitationLSTM(input_dim=9, hidden_dim=256, num_actions=8).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found: {model_path}")
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.eval()
    print(f"Loaded LSTM imitation model from {model_path}")

    # Map country codes to dataset filenames.
    datasets = {
       "NO": "dataset_NO.csv",
       "NL": "dataset_NL.csv",
       "DK": "dataset_DK.csv",
       "FR": "dataset_FR.csv"
    }

    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for country_code, dataset_file in datasets.items():
        if not os.path.exists(dataset_file):
            print(f"Dataset file {dataset_file} not found. Skipping {country_code}.")
            continue
        
        df_result = process_dataset(country_code, dataset_file, lstm_model, device)
        
        country_folder = os.path.join(results_dir, country_code)
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)
        output_csv = os.path.join(country_folder, f"results_hybrid_lstm_{country_code}.csv")
        df_result.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
        # Log per-step metrics to wandb with keys in the specified order.
        wandb.init(project="Thesis_LSTM_location", name=f"{country_code}_validation", reinit=True)
        for i, row in df_result.iterrows():
            log_data = {
                "Bill": row["Bill"],
                "Sell": row["Sell"],
                "Purchase": row["Purchase"],
                "Emissions": row["Emissions"],
                "hydrogen_produced": row["hydrogen_produced"],
                "H2_Purchased_kg": row["H2_Purchased_kg"],
                "H2_Storage": row["H2_Storage"],
                "SoC": row["SoC"],
                "pv_to_load": row["pv_to_load"],
                "battery_to_load": row["battery_to_load"],
                "h2_to_load": row["h2_to_load"],
                "h2_to_load_purchased": row["h2_to_load_purchased"],
                "grid_to_load": row["grid_to_load"],
                "pv_to_battery": row["pv_to_battery"],
                "h2_to_battery": row["h2_to_battery"],
                "grid_to_battery": row["grid_to_battery"],
                "pv_to_grid": row["pv_to_grid"]
            }
            wandb.log(log_data, step=i)
        wandb.finish()

if __name__ == "__main__":
    main()
