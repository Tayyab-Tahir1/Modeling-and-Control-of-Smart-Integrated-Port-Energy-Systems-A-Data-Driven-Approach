# val_islanded_lstm.py

import os
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn

# --- System Parameters (match DQN islanded) ---
# PV rating = 16.2 MW (for normalization only)
BATTERY_CAPACITY     = 8250.0
C_RATE               = 1.0
SOC_MIN              = 0.2 * BATTERY_CAPACITY
SOC_MAX              = 0.8 * BATTERY_CAPACITY
EFFICIENCY           = 0.95

H2_STORAGE_CAPACITY  = 6000.0
ENERGY_PER_KG_H2     = 32.0
FUEL_CELL_EFFICIENCY = 0.5

# Islanded toggle
GRID_AVAILABLE = False  # False = islanded, True = grid‑connected

# Sequence length for LSTM
SEQ_LEN = 24

# === Helpers ===
def compute_max_power(df):
    # remove H2 power cap
    return max(df['Load'].max(), df['PV'].max(), BATTERY_CAPACITY * C_RATE)

def get_feasible_actions(load, pv, soc, h2_storage):
    acts = [0]
    if soc > SOC_MIN + 1e-5:
        acts.append(1)
    if pv > 0:
        acts.append(3)
    if soc < SOC_MAX - 1e-5:
        acts.append(2)
        if GRID_AVAILABLE:
            acts.append(5)
    if h2_storage > 0:
        acts.extend([4,6])
    if GRID_AVAILABLE and load>0:
        acts.append(7)
    return acts

# === LSTM Model Definition ===
class ImitationLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, num_actions=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out)

# === Main ===
if __name__=="__main__":
    run = wandb.init(project="Thesis_Islanded_LSTM_val")

    # Load LSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImitationLSTM().to(device)
    if not os.path.exists("lstm_mpc_imitation.pt"):
        raise FileNotFoundError("lstm_mpc_imitation.pt not found")
    model.load_state_dict(torch.load("lstm_mpc_imitation.pt", map_location=device))
    model.eval()

    # Load data
    df = pd.read_csv("dataset.csv")
    df.rename(columns={'Tou Tariff':'Tou_Tariff','H2 Tariff':'H2_Tariff'}, inplace=True)

    # Pre-allocate columns
    cols = [
      'pv_to_load','pv_to_battery','pv_to_grid',
      'battery_to_load','grid_to_load','grid_to_battery',
      'h2_to_load','hydrogen_produced','h2_to_battery',
      'h2_to_load_purchased','H2_Purchased_kg',
      'Purchase','Sell','Bill','Emissions','SoC','H2_Storage','Chosen_Action'
    ]
    for c in cols:
        df[c]=np.nan

    # Rolling history for the LSTM
    history = []
    soc = BATTERY_CAPACITY * 0.5
    h2_store = 0.0
    max_p = compute_max_power(df)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        L  = row['Load']
        P  = row['PV']
        Tou= row['Tou_Tariff']
        FiT= row['FiT']
        H2r= row['H2_Tariff']
        ef = row.get('Emission_factor',0.5)/1000.0

        # current normalized state
        st = np.array([
            L/max_p,
            P/max_p,
            Tou/df['Tou_Tariff'].max(),
            FiT/df['FiT'].max(),
            H2r/df['H2_Tariff'].max(),
            soc/BATTERY_CAPACITY,
            h2_store/H2_STORAGE_CAPACITY,
            row['Day']/6.0,
            row['Hour']/23.0
        ], dtype=np.float32)

        history.append(st)
        seq = history[-SEQ_LEN:]
        if len(seq)<SEQ_LEN:
            seq = [history[0]]*(SEQ_LEN-len(seq)) + seq
        seq_tensor = torch.tensor([seq], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(seq_tensor)  # (1,SEQ_LEN,8)
            action = int(logits[0,-1,:].argmax().item())

        # if infeasible, random
        feasible = get_feasible_actions(L,P,soc,h2_store)
        if action not in feasible:
            action = int(np.random.choice(feasible))

        # --- flows init ---
        pv_to_load = min(P,L)
        rem_load   = L - pv_to_load
        rem_pv     = P - pv_to_load

        pv_to_battery= pv_to_grid=0.0
        batt_to_load= grid_to_load=grid_to_battery=0.0
        h2_to_load= hydrogen_produced= h2_to_battery= h2_to_load_purchased= H2_Purchased_kg=0.0
        purchase= sell= bill= emission=0.0

        # --- action effects ---
        if action==1:
            avail = (soc-SOC_MIN)*EFFICIENCY
            d = min(rem_load,avail)
            batt_to_load = d
            soc -= d/EFFICIENCY
            rem_load -= d

        elif action==2:
            pv_to_battery = rem_pv
            soc += pv_to_battery*EFFICIENCY
            rem_pv = 0.0

        elif action==3:
            hydrogen_produced = rem_pv/ENERGY_PER_KG_H2
            h2_store = min(h2_store+hydrogen_produced, H2_STORAGE_CAPACITY)
            rem_pv = 0.0

        elif action==4:
            avail_h2 = h2_store*ENERGY_PER_KG_H2
            d = min(rem_load,avail_h2)
            h2_to_load = d
            used = d/ENERGY_PER_KG_H2
            h2_store -= used
            rem_load -= d

        elif action==5 and GRID_AVAILABLE:
            cap = SOC_MAX-soc
            e = cap*0.5
            grid_to_battery = e/EFFICIENCY
            soc += e

        elif action==6:
            avail_h2 = h2_store*ENERGY_PER_KG_H2
            cap = SOC_MAX-soc
            conv = min(avail_h2,cap)
            h2_to_battery = conv*FUEL_CELL_EFFICIENCY
            soc += h2_to_battery
            h2_store -= conv/ENERGY_PER_KG_H2

        elif action==7 and GRID_AVAILABLE:
            H2_Purchased_kg = rem_load/(ENERGY_PER_KG_H2*FUEL_CELL_EFFICIENCY)
            h2_to_load_purchased = rem_load
            purchase = H2_Purchased_kg*H2r
            rem_load=0.0

        # islanded: leftover → H2
        if not GRID_AVAILABLE and rem_load>0:
            use = min(h2_store*ENERGY_PER_KG_H2, rem_load)
            h2_to_load += use
            h2_store -= use/ENERGY_PER_KG_H2
            rem_load -= use
            if rem_load>1e-6:
                req = rem_load/(ENERGY_PER_KG_H2*FUEL_CELL_EFFICIENCY)
                h2_to_load_purchased += rem_load
                H2_Purchased_kg = req
                purchase = req*H2r
                rem_load=0.0

        # PV export / grid import
        if GRID_AVAILABLE:
            pv_to_grid   = rem_pv
            grid_to_load = rem_load
        else:
            pv_to_grid = grid_to_load = grid_to_battery = 0.0
            sell = 0.0

        # financials
        if GRID_AVAILABLE:
            grid_cost = (grid_to_load+grid_to_battery)*Tou
            sell = pv_to_grid*FiT
            bill = grid_cost - sell + purchase
            emission = (grid_to_load+grid_to_battery)*ef
        else:
            bill = purchase
            # emission = 0.0
            emission = (grid_to_load+grid_to_battery)*ef

        # clamp SOC
        soc = max(SOC_MIN, min(soc, SOC_MAX))

        # record
        df.at[idx,'pv_to_load']=pv_to_load
        df.at[idx,'pv_to_battery']=pv_to_battery
        df.at[idx,'pv_to_grid']=pv_to_grid
        df.at[idx,'battery_to_load']=batt_to_load
        df.at[idx,'grid_to_load']=grid_to_load
        df.at[idx,'grid_to_battery']=grid_to_battery
        df.at[idx,'h2_to_load']=h2_to_load
        df.at[idx,'hydrogen_produced']=hydrogen_produced
        df.at[idx,'h2_to_battery']=h2_to_battery
        df.at[idx,'h2_to_load_purchased']=h2_to_load_purchased
        df.at[idx,'H2_Purchased_kg']=H2_Purchased_kg
        df.at[idx,'Purchase']=purchase
        df.at[idx,'Sell']=sell
        df.at[idx,'Bill']=bill
        df.at[idx,'Emissions']=emission
        df.at[idx,'SoC']=(soc/BATTERY_CAPACITY)*100.0
        df.at[idx,'H2_Storage']=(h2_store/H2_STORAGE_CAPACITY)*100.0
        df.at[idx,'Chosen_Action']=action

        # W&B line logs
        wandb.log({
            'pv_to_load':pv_to_load,
            'pv_to_battery':pv_to_battery,
            'battery_to_load':batt_to_load,
            'h2_to_load':h2_to_load,
            'hydrogen_produced':hydrogen_produced,
            'h2_to_battery':h2_to_battery,
            'h2_to_load_purchased':h2_to_load_purchased,
            'H2_Purchased_kg':H2_Purchased_kg,
            'Purchase':purchase,
            'Sell':sell,
            'Bill':bill,
            'Emissions':emission,
            'SoC':(soc/BATTERY_CAPACITY)*100.0,
            'H2_Storage':(h2_store/H2_STORAGE_CAPACITY)*100.0
        }, step=idx)

    # save and finish
    out_csv = "islanded_validation_lstm.csv"
    df.to_csv(out_csv, index=False)
    wandb.save(out_csv)
    run.finish()
