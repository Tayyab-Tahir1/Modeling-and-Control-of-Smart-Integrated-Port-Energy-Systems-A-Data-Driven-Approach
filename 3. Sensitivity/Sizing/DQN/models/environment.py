import numpy as np
import pandas as pd

class EnergyEnv:
    def __init__(
        self,
        data: pd.DataFrame,
        pv_col: str = "PV",
        battery_capacity: float = 5500.0,
        h2_storage_capacity: float = 2000.0
    ):
        # Store input data and PV column selection
        self.data = data.reset_index(drop=True)
        self.pv_col = pv_col
        self.max_steps = len(self.data)

        # ---- Resource Capacities ----
        # Battery
        self.battery_capacity = battery_capacity  # kWh
        self.c_rate = 0.5
        self.max_charge_discharge_power = self.c_rate * self.battery_capacity  # kW reference
        self.soc_min = 0.2 * self.battery_capacity
        self.soc_max = 0.8 * self.battery_capacity
        self.efficiency = 0.95  # round-trip

        # Hydrogen storage
        self.h2_storage_capacity = h2_storage_capacity  # kg
        self.h2_storage = 0.0  # kg
        self.h2_capacity = 2000.0  # kW reference for conversion
        self.energy_per_kg_H2 = 32.0  # kWh per kg

        # ---- Emission & Cost Weights ----
        if "Emission_factor" in self.data.columns:
            self.default_emission_factor = None
        else:
            self.default_emission_factor = 0.5  # g/kWh
        self.cost_weight = 1.0
        self.emission_weight = 0.2
        self.grid_charge_fraction = 0.5
        self.fuel_cell_efficiency = 0.5

        # ---- Normalization Values ----
        self.load_max = self.data['Load'].max()
        self.pv_max = self.data[self.pv_col].max()
        self.tou_max = self.data['Tou_Tariff'].max()
        self.fit_max = self.data['FiT'].max()
        self.h2_max = self.data['H2_Tariff'].max()
        self.max_power = max(
            self.load_max,
            self.pv_max,
            self.max_charge_discharge_power,
            self.h2_capacity
        )

        # Initialize state and print summary
        self.reset()
        self._print_summary()

    def _print_summary(self):
        print("System Configuration:")
        print(f"  PV Column: {self.pv_col}, PV Max: {self.pv_max:.2f} kW")
        print(f"  Battery Capacity: {self.battery_capacity:.2f} kWh")
        print(f"  H₂ Storage Capacity: {self.h2_storage_capacity:.2f} kg")
        print(f"  Battery Power Ref: {self.max_charge_discharge_power:.2f} kW")
        print(f"  H₂ Power Ref: {self.h2_capacity:.2f} kW")
        print(f"  Composite Weights: Cost={self.cost_weight}, Emission={self.emission_weight}\n")

    def reset(self):
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False
        self.soc = 0.5 * self.battery_capacity
        self.h2_storage = 0.0
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        pv = row[self.pv_col]
        state = np.array([
            row['Load'] / self.max_power,
            pv / self.max_power,
            row['Tou_Tariff'] / self.tou_max,
            row['FiT'] / self.fit_max,
            row['H2_Tariff'] / self.h2_max,
            self.soc / self.battery_capacity,
            self.h2_storage / self.h2_storage_capacity,
            row['Day'] / 6.0,
            row['Hour'] / 23.0
        ], dtype=np.float32)
        return state

    def get_feasible_actions(self):
        row = self.data.iloc[self.current_step]
        pv = row[self.pv_col]
        feasible = [0]
        if self.soc > self.soc_min + 1e-5:
            feasible.append(1)
        if self.soc < self.soc_max - 1e-5:
            feasible.extend([2, 5])
        if pv > 0:
            feasible.append(3)
        if self.h2_storage > 0:
            feasible.extend([4, 6])
        if row['Load'] > 0:
            feasible.append(7)
        return feasible

    def step(self, action: int):
        if self.done:
            return np.zeros(9, dtype=np.float32), 0.0, True, {}

        row = self.data.iloc[self.current_step]
        load = row['Load']
        pv = row[self.pv_col]
        tou = row['Tou_Tariff']
        fit = row['FiT']
        h2_tariff = row['H2_Tariff']

        # PV direct to load
        pv_to_load = min(pv, load)
        load_rem = load - pv_to_load
        pv_rem = pv - pv_to_load

        # Initialize flows
        pv_to_batt = pv_to_grid = batt_to_load = grid_to_load = grid_to_batt = 0.0
        h2_to_load = hyd_prod = h2_to_batt = h2_load_pur = 0.0
        H2_purchase_cost = 0.0
        req_kg = 0.0

        # Action effects
        if action == 1:
            avail_e = (self.soc - self.soc_min) * self.efficiency
            batt_to_load = min(load_rem, avail_e)
            self.soc -= batt_to_load / self.efficiency
            load_rem -= batt_to_load
        elif action == 2:
            pv_to_batt = pv_rem
            self.soc += pv_to_batt * self.efficiency
            pv_rem = 0.0
        elif action == 3:
            hyd_prod = pv_rem / self.energy_per_kg_H2
            self.h2_storage = min(self.h2_storage + hyd_prod, self.h2_storage_capacity)
            pv_rem = 0.0
        elif action == 4:
            avail_h2_e = self.h2_storage * self.energy_per_kg_H2
            h2_to_load = min(load_rem, avail_h2_e)
            used_kg = h2_to_load / self.energy_per_kg_H2
            self.h2_storage -= used_kg
            load_rem -= h2_to_load
        elif action == 5:
            cap_rem = self.soc_max - self.soc
            grid_to_batt = cap_rem * self.grid_charge_fraction / self.efficiency
            self.soc += cap_rem * self.grid_charge_fraction
        elif action == 6:
            avail_h2_e = self.h2_storage * self.energy_per_kg_H2
            batt_cap_rem = self.soc_max - self.soc
            conv_e = min(avail_h2_e, batt_cap_rem)
            h2_to_batt = conv_e * self.fuel_cell_efficiency
            self.soc += h2_to_batt
            self.h2_storage -= conv_e / self.energy_per_kg_H2
        elif action == 7:
            req_kg = load_rem / (self.energy_per_kg_H2 * self.fuel_cell_efficiency)
            h2_load_pur = load_rem
            H2_purchase_cost = req_kg * h2_tariff
            load_rem = 0.0

        grid_to_load = load_rem
        pv_to_grid = pv_rem
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)

        grid_cost = (grid_to_load + grid_to_batt) * tou
        pv_revenue = pv_to_grid * fit
        bill = grid_cost - pv_revenue + H2_purchase_cost
        ef = (row.get('Emission_factor', None) / 1000.0
              if 'Emission_factor' in row else self.default_emission_factor / 1000.0)
        emissions = (grid_to_load + grid_to_batt) * ef

        composite = self.cost_weight * bill + self.emission_weight * emissions
        max_bill = self.max_power * max(self.tou_max, self.h2_max) or 1.0
        reward = - composite / max_bill

        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        next_state = self._get_state() if not self.done else np.zeros(9, dtype=np.float32)

        info = {
            'pv_to_load': pv_to_load,
            'pv_to_battery': pv_to_batt,
            'pv_to_grid': pv_to_grid,
            'battery_to_load': batt_to_load,
            'grid_to_load': grid_to_load,
            'grid_to_battery': grid_to_batt,
            'h2_to_load': h2_to_load,
            'hydrogen_produced': hyd_prod,
            'h2_to_battery': h2_to_batt,
            'h2_to_load_purchased': h2_load_pur,
            'H2_Purchased_kg': req_kg,
            'Purchase': grid_cost,
            'Sell': pv_revenue,
            'Bill': bill,
            'Emissions': emissions,
            'SoC': (self.soc / self.battery_capacity) * 100.0,
            'H2_Storage': (self.h2_storage / self.h2_storage_capacity) * 100.0
        }
        return next_state, reward, self.done, info

    def state_size(self):
        return 9
