import numpy as np
import pandas as pd

class EnergyEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.max_steps = len(data)
        
        # Battery specifications
        self.battery_capacity = 5500  # kWh
        self.c_rate = 0.5
        # (max_charge_discharge_power is used only for normalization reference)
        self.max_charge_discharge_power = self.c_rate * self.battery_capacity  
        self.soc_min = 0.2 * self.battery_capacity  # 20% SoC
        self.soc_max = 0.8 * self.battery_capacity  # 80% SoC
        self.efficiency = 0.95  # Charging/discharging efficiency
        
        # Hydrogen system specifications (for reference)
        self.h2_capacity = 2000  # kW (reference value)
        # Hydrogen storage (in kg)
        self.h2_storage = 0.0         
        self.h2_storage_capacity = 3000.0  # kg
        
        # Energy required per kg of H2 produced (electrolysis conversion factor)
        self.energy_per_kg_H2 = 32  # kWh/kg
        
        # Emission factor for grid imports.
        # If the input data contains an "Emission_factor" column (in gram/kWh), we use it per step.
        # Otherwise, we default to 0.5 g/kWh.
        if "Emission_factor" in data.columns:
            self.emission_factor = None  # will be read from each row in step()
        else:
            self.emission_factor = 0.5  # default (in gram/kWh)
        
        # New: Cost weight and emission weight for composite metric.
        self.cost_weight = 1.0      # Weight for cost (bill)
        self.emission_weight = 0.2  # Lower weight for emissions
        
        # Parameter for grid battery charging:
        self.grid_charge_fraction = 0.5  # Fraction of available battery capacity used when charging from grid
        
        # Fuel cell conversion efficiency (for converting H₂ to electrical energy)
        self.fuel_cell_efficiency = 0.5
        
        # Calculate maximum possible power for normalization
        self.max_power = max(
            data['Load'].max(),
            data['PV'].max(),
            self.max_charge_discharge_power,
            self.h2_capacity
        )
        
        # Store maximum values from the dataset (for state normalization)
        self.load_max = data['Load'].max()
        self.pv_max = data['PV'].max()
        self.tou_max = data['Tou_Tariff'].max()
        self.fit_max = data['FiT'].max()
        self.h2_max = data['H2_Tariff'].max()
        
        # Print summary info. If the data has an Emission_factor column, print its summary.
        print("System Capacities:")
        print(f"Battery Capacity: {self.battery_capacity:.2f} kWh")
        print(f"Battery Power (ref): {self.max_charge_discharge_power:.2f} kW")
        print(f"H2 System Capacity (ref): {self.h2_capacity:.2f} kW")
        print("\nMaximum Values:")
        print(f"Max Power: {self.max_power:.2f} kW")
        print(f"Max Load: {self.load_max:.2f} kW")
        print(f"Max PV: {self.pv_max:.2f} kW")
        print(f"Max ToU Tariff: {self.tou_max:.4f}")
        print(f"Max FiT: {self.fit_max:.4f}")
        print(f"Max H2 Tariff: {self.h2_max:.4f}")
        if "Emission_factor" in data.columns:
            print(f"Emission Factor (from data): {data['Emission_factor'].mean():.2f} g/kWh (mean)")
        else:
            print(f"Emission Factor (default): {self.emission_factor:.2f} g/kWh")
        print(f"Energy per kg H2: {self.energy_per_kg_H2:.2f} kWh/kg")
        print(f"Cost Weight: {self.cost_weight}, Emission Weight: {self.emission_weight}")
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.done = False
        # Initialize battery at 50% SoC and empty hydrogen storage.
        self.soc = self.battery_capacity * 0.5  
        self.h2_storage = 0.0  
        return self._get_state()
    
    def _get_state(self):
        row = self.data.iloc[self.current_step]
        # State vector: 9 dimensions
        state = np.array([
            row['Load'] / self.max_power,
            row['PV'] / self.max_power,
            row['Tou_Tariff'] / self.tou_max,
            row['FiT'] / self.fit_max,
            row['H2_Tariff'] / self.h2_max,
            self.soc / self.battery_capacity,
            self.h2_storage / self.h2_storage_capacity,
            row['Day'] / 6,
            row['Hour'] / 23
        ])
        return state

    def get_feasible_actions(self):
        """
        Expanded action space:
          0: Do nothing
          1: Battery discharge to meet load
          2: Charge battery using excess PV
          3: Produce hydrogen using excess PV (store produced H2)
          4: Use stored hydrogen directly to meet load
          5: Charge battery using grid power
          6: Convert stored hydrogen to battery energy (H2-to-Battery)
          7: Purchase hydrogen from grid to meet load
        """
        row = self.data.iloc[self.current_step]
        feasible = [0]
        if self.soc > self.soc_min + 1e-5:
            feasible.append(1)
        if self.soc < self.soc_max - 1e-5:
            feasible.append(2)
            feasible.append(5)
        if row['PV'] > 0:
            feasible.append(3)
        if self.h2_storage > 0:
            feasible.append(4)
            feasible.append(6)
        if row['Load'] > 0:
            feasible.append(7)
        return feasible

    def step(self, action):
        if self.done:
            info = {
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
                'Purchase': 0.0,
                'Sell': 0.0,
                'Bill': 0.0,
                'Emissions': 0.0,
                'SoC': self.soc / self.battery_capacity * 100,
                'H2_Storage': self.h2_storage / self.h2_storage_capacity * 100,
                'System_Max_Power': self.max_power
            }
            return np.zeros(9), 0, self.done, info

        try:
            row = self.data.iloc[self.current_step]
        except IndexError:
            print(f"Error: current_step {self.current_step} is out of bounds.")
            return np.zeros(9), 0, True, {}
        
        # Retrieve current values.
        load = row['Load']
        pv = row['PV']
        tou_tariff = row['Tou_Tariff']
        fit = row['FiT']
        h2_tariff = row['H2_Tariff']
        
        # Use PV to directly meet load.
        pv_to_load = min(pv, load)
        load_remaining = load - pv_to_load
        pv_remaining = pv - pv_to_load
        
        # Initialize flow variables.
        pv_to_battery = 0.0
        pv_to_grid = 0.0
        battery_to_load = 0.0
        grid_to_load = 0.0
        grid_to_battery = 0.0
        h2_to_load = 0.0
        hydrogen_produced = 0.0
        h2_to_battery = 0.0
        h2_to_load_purchased = 0.0
        
        H2_purchase_cost = 0  # Cost incurred if hydrogen is purchased
        
        fc_eff = self.fuel_cell_efficiency
        
        # Process chosen action:
        if action == 0:
            pass
        elif action == 1:
            available_energy = (self.soc - self.soc_min) * self.efficiency
            battery_to_load = min(load_remaining, available_energy)
            self.soc -= battery_to_load / self.efficiency
            load_remaining -= battery_to_load
        elif action == 2:
            pv_to_battery = pv_remaining
            self.soc += pv_to_battery * self.efficiency
            pv_remaining = 0
        elif action == 3:
            energy_used = pv_remaining
            hydrogen_produced = energy_used / self.energy_per_kg_H2
            self.h2_storage = min(self.h2_storage + hydrogen_produced, self.h2_storage_capacity)
            pv_remaining = 0
        elif action == 4:
            available_h2_energy = self.h2_storage * self.energy_per_kg_H2
            h2_to_load = min(load_remaining, available_h2_energy)
            hydrogen_used = h2_to_load / self.energy_per_kg_H2
            self.h2_storage -= hydrogen_used
            load_remaining -= h2_to_load
        elif action == 5:
            available_capacity = self.soc_max - self.soc
            energy_to_charge = available_capacity * self.grid_charge_fraction
            grid_to_battery = energy_to_charge / self.efficiency
            self.soc += energy_to_charge
        elif action == 6:
            available_h2_energy = self.h2_storage * self.energy_per_kg_H2
            battery_capacity_remaining = self.soc_max - self.soc
            energy_converted = min(available_h2_energy, battery_capacity_remaining)
            battery_energy_gained = energy_converted * fc_eff
            h2_to_battery = battery_energy_gained
            self.soc += battery_energy_gained
            hydrogen_used = energy_converted / self.energy_per_kg_H2
            self.h2_storage -= hydrogen_used
        elif action == 7:
            hydrogen_required = load_remaining / (self.energy_per_kg_H2 * fc_eff)
            h2_to_load_purchased = load_remaining
            H2_purchase_cost = hydrogen_required * h2_tariff
            load_remaining = 0
        
        grid_to_load = load_remaining
        pv_to_grid = pv_remaining

        self.soc = max(self.soc_min, min(self.soc, self.soc_max))
        
        grid_cost = (grid_to_load + grid_to_battery) * tou_tariff
        pv_revenue = pv_to_grid * fit
        bill = grid_cost - pv_revenue + H2_purchase_cost
        
        # Use emission factor from dataset if available; otherwise use default.
        if "Emission_factor" in row:
            # row['Emission_factor'] is in grams per kWh; convert to kg per kWh.
            emission_factor = row['Emission_factor'] / 1000.0
        else:
            emission_factor = self.emission_factor / 1000.0
        
        emissions = (grid_to_load + grid_to_battery) * emission_factor
        
        composite = (self.cost_weight * bill) + (self.emission_weight * emissions)
        max_possible_bill = self.max_power * max(self.tou_max, self.h2_max)
        if max_possible_bill == 0:
            max_possible_bill = 1
        reward = - composite / max_possible_bill
        
        self.total_reward += reward
        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        next_state = self._get_state() if not self.done else np.zeros(9)
        
        info = {
            'pv_to_load': pv_to_load,
            'pv_to_battery': pv_to_battery,
            'pv_to_grid': pv_to_grid,
            'battery_to_load': battery_to_load,
            'grid_to_load': grid_to_load,
            'grid_to_battery': grid_to_battery,
            'h2_to_load': h2_to_load,
            'hydrogen_produced': hydrogen_produced,
            'h2_to_battery': h2_to_battery,
            'h2_to_load_purchased': h2_to_load_purchased,
            'H2_Purchased_kg': (hydrogen_required if action == 7 else 0),
            'Purchase': grid_cost,
            'Sell': pv_revenue,
            'Bill': bill,
            'Emissions': emissions,
            'SoC': self.soc / self.battery_capacity * 100,
            'H2_Storage': self.h2_storage / self.h2_storage_capacity * 100,
            'System_Max_Power': self.max_power
        }
        
        return next_state, reward, self.done, info

    def state_size(self):
        return 9
