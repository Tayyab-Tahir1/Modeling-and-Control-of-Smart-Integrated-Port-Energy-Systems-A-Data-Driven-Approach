import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb  # For experiment tracking
import pickle

# Import MPC functions and constants
from MPC_model import mpc_step, compute_max_power, BATTERY_CAPACITY, H2_STORAGE_CAPACITY, ENERGY_PER_KG_H2, HORIZON

# ------------------------------------------------
# 1) Helper: Map MPC flows to a discrete action
# ------------------------------------------------
def get_expert_action(flows):
    """
    Given the MPC flows dictionary, choose the discrete action (0-7) that best represents the expert decision.
    """
    threshold = 1e-3
    action_values = {
        1: flows.get('battery_to_load', 0),
        2: flows.get('pv_to_battery', 0),
        3: flows.get('hydrogen_produced', 0) * ENERGY_PER_KG_H2,  # convert kg to kWh
        4: flows.get('h2_to_load', 0),
        5: flows.get('grid_to_battery', 0),
        6: flows.get('h2_to_battery', 0),
        7: flows.get('h2_to_load_purchased', 0)
    }
    max_action = 0
    max_val = 0.0
    for act, val in action_values.items():
        if val > max_val and val > threshold:
            max_val = val
            max_action = act
    return max_action

# ------------------------------------------------
# 2) Sequence Dataset Builder
# ------------------------------------------------
def build_finetuning_sequences(df, seq_len=24):
    """
    Collect short sequences from the dataset to mimic MPC's horizon-based decisions.
    Each sample is a sequence of states and the corresponding sequence of expert actions.
    """
    soc = BATTERY_CAPACITY * 0.5
    h2_storage = 0.0

    max_power = compute_max_power(df)
    max_tou = df['Tou_Tariff'].max()
    max_fit = df['FiT'].max()
    max_h2_tariff = df['H2_Tariff'].max()

    N = len(df)
    sequences = []

    for i in tqdm(range(N - seq_len), desc="Building expert sequences"):
        if i == 0:
            soc = BATTERY_CAPACITY * 0.5
            h2_storage = 0.0

        states_seq = []
        actions_seq = []

        for t in range(i, i + seq_len):
            if t + HORIZON > N:
                break

            row = df.iloc[t]
            load = row['Load']
            pv = row['PV']
            tou_tariff = row['Tou_Tariff']
            fit = row['FiT']
            h2_tariff = row['H2_Tariff']
            day = row['Day']
            hour = row['Hour']

            state = np.array([
                load / max_power,
                pv / max_power,
                tou_tariff / max_tou,
                fit / max_fit,
                h2_tariff / max_h2_tariff,
                soc / BATTERY_CAPACITY,
                h2_storage / H2_STORAGE_CAPACITY,
                day / 6.0,
                hour / 23.0
            ], dtype=np.float32)

            forecast = df.iloc[t : t + HORIZON].copy()
            mpc_out = mpc_step(forecast, soc, h2_storage)
            if mpc_out is None:
                expert_action = 0
            else:
                flows = mpc_out.get('flows', {})
                expert_action = get_expert_action(flows)
                soc = mpc_out.get('SoC', soc)
                h2_storage = mpc_out.get('H2_Storage', h2_storage)

            states_seq.append(state)
            actions_seq.append(expert_action)

        if len(states_seq) == seq_len:
            sequences.append((np.array(states_seq), np.array(actions_seq)))

    return sequences

# ------------------------------------------------
# 3) LSTM Imitation Model
# ------------------------------------------------
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

# ------------------------------------------------
# 4) Training Function with LR Scheduler and Early Stopping
# ------------------------------------------------
def train_lstm(model, sequences, initial_epochs, batch_size, lr, patience=20, max_epochs=5000):
    """
    Train the LSTM model on sequence data using cross-entropy loss.
    Training will run for at least 'initial_epochs'. If the average loss does not improve 
    for 'patience' consecutive epochs, training stops. Otherwise, training can continue up to 'max_epochs'.
    A ReduceLROnPlateau scheduler is used with the same patience.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.1, patience=patience,
                                                     verbose=True)

    all_states = []
    all_actions = []
    for (states_seq, actions_seq) in sequences:
        all_states.append(states_seq)
        all_actions.append(actions_seq)

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)

    all_states_t = torch.tensor(all_states, dtype=torch.float32).to(device)
    all_actions_t = torch.tensor(all_actions, dtype=torch.long).to(device)

    dataset_size = all_states_t.shape[0]
    num_batches = int(np.ceil(dataset_size / batch_size))

    best_loss = float('inf')
    early_stop_counter = 0
    epoch = 0

    while epoch < initial_epochs or (epoch < max_epochs and early_stop_counter < patience):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", leave=False)
        for i in pbar:
            idx = permutation[i * batch_size : (i + 1) * batch_size]
            batch_states = all_states_t[idx]
            batch_actions = all_actions_t[idx]

            optimizer.zero_grad()
            logits = model(batch_states)

            logits_2d = logits.reshape(-1, logits.size(-1))
            actions_1d = batch_actions.reshape(-1)

            loss = criterion(logits_2d, actions_1d)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        wandb.log({"epoch": epoch+1, "average_loss": avg_loss, "learning_rate": optimizer.param_groups[0]['lr']})
        epoch += 1

    print(f"Training stopped at epoch {epoch} with best loss {best_loss:.6f}")
    return model

# ------------------------------------------------
# 5) Main Script
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sequence-based LSTM imitation of MPC decisions")
    parser.add_argument("--dataset", type=str, default="dataset.csv", help="Path to dataset CSV file")
    parser.add_argument("--seq_len", type=int, default=24, help="Length of each training sequence (hours)")
    parser.add_argument("--initial_epochs", type=int, default=1000, help="Minimum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Patience for LR scheduler and early stopping")
    parser.add_argument("--max_epochs", type=int, default=5000, help="Maximum number of training epochs")
    parser.add_argument("--output_model", type=str, default="lstm_mpc_imitation.pt", help="Where to save the trained model")
    args = parser.parse_args()

    # Set wandb to offline mode
    wandb.init(project="MPC_Imitation_LSTM", config=vars(args), mode="offline")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    df = pd.read_csv(args.dataset)
    df.rename(columns={'Tou Tariff': 'Tou_Tariff', 'H2 Tariff': 'H2_Tariff'}, inplace=True)

    # Precompute expert sequences or load them if they already exist using pickle.
    expert_data_file = "expert_sequences.pkl"
    if os.path.exists(expert_data_file):
        print("Loading precomputed expert sequences...")
        with open(expert_data_file, "rb") as f:
            expert_sequences = pickle.load(f)
    else:
        print("Building expert sequences with MPC. This may take some time...")
        expert_sequences = build_finetuning_sequences(df, seq_len=args.seq_len)
        with open(expert_data_file, "wb") as f:
            pickle.dump(expert_sequences, f)
        print(f"Expert sequences saved to {expert_data_file}")

    print(f"Collected {len(expert_sequences)} sequences of length {args.seq_len}.")

    print("Initializing LSTM model...")
    model = ImitationLSTM(input_dim=9, hidden_dim=256, num_actions=8)

    print("Training LSTM on expert data...")
    model = train_lstm(model, expert_sequences, initial_epochs=args.initial_epochs,
                       batch_size=args.batch_size, lr=args.lr,
                       patience=args.patience, max_epochs=args.max_epochs)

    print("Saving trained model...")
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved as {args.output_model}")
    wandb.save(args.output_model)

if __name__ == "__main__":
    main()
