"""Real-world continuous energy storage optimization for electricity arbitrage."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import requests
from scipy.interpolate import interp1d
from dotenv import load_dotenv
import os
import seaborn as sns

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyArbitrageOptimizer:
    """Optimize battery storage operation for continuous electricity price arbitrage."""

    def __init__(
            self,
            battery_capacity_kwh: float = 100.0,
            max_power_kw: float = 20.0,
            battery_efficiency: float = 0.92,
            initial_soc: float = 0.3,
            min_soc: float = 0.1,
            max_soc: float = 0.9,
            cycle_life: int = 3000,            # Battery cycle life
            calendar_life_years: float = 10.0,  # Calendar life in years
            battery_cost: float = 400.0,        # Cost per kWh of battery capacity
            time_resolution: str = '15min'      # Time resolution for optimization
    ) -> None:
        """Initialize with realistic battery parameters."""
        self.battery_capacity = battery_capacity_kwh
        self.max_power = max_power_kw
        self.efficiency = np.sqrt(battery_efficiency)
        self.initial_soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.time_resolution = pd.Timedelta(time_resolution)
        
        # Advanced battery parameters
        self.cycle_life = cycle_life
        self.calendar_life_years = calendar_life_years
        self.battery_cost = battery_cost
        
        # Calculate degradation cost ($/kWh cycled)
        self.degradation_cost = (battery_cost * battery_capacity_kwh) / (cycle_life * battery_capacity_kwh * 2)
        
        # Temperature derating factors (example values)
        self.temp_derating = {
            'power': {
                'low': 0.7,   # Below 0°C
                'normal': 1.0, # 0-40°C
                'high': 0.8    # Above 40°C
            },
            'capacity': {
                'low': 0.8,
                'normal': 1.0,
                'high': 0.9
            }
        }
        
        self.model = None
        self.results = None
        self.price_data = None
        self.optimization_window = timedelta(days=2)  # Rolling optimization window
        
        # Create cache directory
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)

    def get_price_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get real electricity price data with fallback to synthetic data.
        Attempts to load from cache first, then API, then generates synthetic.
        """
        cache_file = self.cache_dir / f"prices_{start_date.date()}_{end_date.date()}.parquet"
        
        # Try loading from cache
        if cache_file.exists():
            return pd.read_parquet(cache_file)
            
        try:
            # Example API call (replace with actual API endpoint)
            prices = self._fetch_price_data_from_api(start_date, end_date)
        except Exception as e:
            logger.warning(f"Failed to fetch real price data: {e}. Using synthetic data.")
            prices = self._generate_synthetic_prices(start_date, end_date)
            
        # Cache the data
        prices.to_parquet(cache_file)
        return prices

    def _fetch_price_data_from_api(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch real-time electricity prices from an API."""
        # Example using EIA API (you'll need to implement actual API calls)
        api_url = "https://api.eia.gov/series/"
        params = {
            "api_key": os.getenv("API_KEY"),
            "series_id": "ELEC.REAL.TIME.PRICE",
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        }
        
        response = requests.get(api_url, params=params)
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}")
            
        # Process API response (implement actual data processing)
        data = response.json()
        # Convert to DataFrame...
        
        return pd.DataFrame()  # Placeholder

    def _generate_synthetic_prices(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate realistic synthetic price data with multiple components."""
        periods = pd.date_range(start_date, end_date, freq=self.time_resolution)
        
        def base_load_pattern(hour):
            return 0.3 + 0.2 * np.sin(2 * np.pi * (hour - 6) / 24)
            
        def peak_pattern(hour):
            morning_peak = 0.2 * np.exp(-((hour - 9) ** 2) / 4)
            evening_peak = 0.3 * np.exp(-((hour - 19) ** 2) / 4)
            return morning_peak + evening_peak
            
        def seasonal_pattern(date):
            day_of_year = date.dayofyear
            return 0.1 * np.sin(2 * np.pi * day_of_year / 365)
            
        prices = []
        for dt in periods:
            hour_float = dt.hour + dt.minute / 60.0
            
            # Combine multiple price components
            base_price = base_load_pattern(hour_float)
            peak_price = peak_pattern(hour_float)
            seasonal_adj = seasonal_pattern(dt)
            
            # Add random variations (using smooth noise)
            noise = 0.05 * np.sin(2 * np.pi * hour_float / 1.5 + np.random.rand())
            
            # Combine all components
            price = base_price + peak_price + seasonal_adj + noise
            
            # Add occasional price spikes
            if np.random.random() < 0.01:  # 1% chance of price spike
                price *= np.random.uniform(1.5, 3.0)
                
            prices.append(max(0.10, price))  # Ensure minimum price

        return pd.DataFrame({
            'datetime': periods,
            'price': prices
        })

    def optimize(self, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Optimize battery operation over a time window."""
        if start_date is None:
            start_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        if end_date is None:
            end_date = start_date + self.optimization_window
            
        # Get price data
        self.price_data = self.get_price_data(start_date, end_date)
        
        # Build and solve model
        self.model = self._build_model()
        return self._solve_model()

    def _build_model(self) -> pyo.ConcreteModel:
        """Build the optimization model with realistic constraints."""
        model = pyo.ConcreteModel("Energy Arbitrage")

        # Sets and Parameters
        model.T = pyo.Set(initialize=range(len(self.price_data)), ordered=True)
        model.prices = pyo.Param(model.T, initialize=dict(enumerate(self.price_data['price'])))
        model.dt = pyo.Param(initialize=self.time_resolution.total_seconds() / 3600)  # Time step in hours

        # Variables
        model.charge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.max_power))
        model.discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.max_power))
        model.soc = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(self.min_soc, self.max_soc))

        # Objective function with realistic components
        def objective_rule(mdl):
            # Revenue from discharge minus cost of charging
            revenue = sum(mdl.dt * (mdl.prices[t] * mdl.discharge[t] - mdl.prices[t] * mdl.charge[t]) 
                         for t in mdl.T)
            
            # Degradation costs
            cycling_cost = sum(mdl.dt * (mdl.discharge[t] + mdl.charge[t]) 
                              for t in mdl.T) * self.degradation_cost
            
            # Remove the nonlinear switching penalty
            return revenue - cycling_cost

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        # Enhanced constraints
        def soc_dynamics_rule(mdl, t):
            if t == 0:
                return mdl.soc[t] == self.initial_soc + mdl.dt * (
                    self.efficiency * mdl.charge[t] - mdl.discharge[t] / self.efficiency
                ) / self.battery_capacity
            return mdl.soc[t] == mdl.soc[t-1] + mdl.dt * (
                self.efficiency * mdl.charge[t] - mdl.discharge[t] / self.efficiency
            ) / self.battery_capacity

        model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)

        # Ramping constraints
        def ramp_up_charge_rule(mdl, t):
            if t == 0:
                return pyo.Constraint.Skip
            max_ramp = self.max_power * 0.5
            return mdl.charge[t] - mdl.charge[t-1] <= max_ramp

        def ramp_down_charge_rule(mdl, t):
            if t == 0:
                return pyo.Constraint.Skip
            max_ramp = self.max_power * 0.5
            return mdl.charge[t] - mdl.charge[t-1] >= -max_ramp

        def ramp_up_discharge_rule(mdl, t):
            if t == 0:
                return pyo.Constraint.Skip
            max_ramp = self.max_power * 0.5
            return mdl.discharge[t] - mdl.discharge[t-1] <= max_ramp

        def ramp_down_discharge_rule(mdl, t):
            if t == 0:
                return pyo.Constraint.Skip
            max_ramp = self.max_power * 0.5
            return mdl.discharge[t] - mdl.discharge[t-1] >= -max_ramp

        # Add each ramp constraint separately
        model.ramp_up_charge = pyo.Constraint(model.T, rule=ramp_up_charge_rule)
        model.ramp_down_charge = pyo.Constraint(model.T, rule=ramp_down_charge_rule)
        model.ramp_up_discharge = pyo.Constraint(model.T, rule=ramp_up_discharge_rule)
        model.ramp_down_discharge = pyo.Constraint(model.T, rule=ramp_down_discharge_rule)

        return model

    def _solve_model(self) -> bool:
        """Solve the optimization model with enhanced error handling."""
        try:
            # Use GLPK solver with no additional options
            solver = pyo.SolverFactory('glpk')
            
            # Simple solve call without any options
            self.results = solver.solve(self.model, tee=True)  # Set tee=True to see solver output
            
            if (self.results.solver.status == SolverStatus.ok and 
                self.results.solver.termination_condition == TerminationCondition.optimal):
                return True
                
            logger.warning(f"Solver Status: {self.results.solver.status}")
            return False
            
        except Exception as exc:
            logger.error(f"Error solving model: {str(exc)}")
            return False

    def get_results(self) -> Optional[Dict]:
        """Get optimization results with additional metrics."""
        if not self.results or self.price_data.empty:
            return None

        # Basic results
        results = {
            'datetime': self.price_data['datetime'].tolist(),
            'prices': self.price_data['price'].tolist(),
            'charge': [pyo.value(self.model.charge[t]) for t in self.model.T],
            'discharge': [pyo.value(self.model.discharge[t]) for t in self.model.T],
            'soc': [pyo.value(self.model.soc[t]) for t in self.model.T]
        }
        
        # Calculate additional metrics
        dt_hours = self.time_resolution.total_seconds() / 3600
        
        # Energy metrics
        results['total_charged'] = sum(c * dt_hours for c in results['charge'])
        results['total_discharged'] = sum(d * dt_hours for d in results['discharge'])
        results['round_trip_efficiency'] = results['total_discharged'] / results['total_charged']
        
        # Financial metrics
        results['revenue'] = sum(p * d * dt_hours for p, d in zip(results['prices'], results['discharge']))
        results['cost'] = sum(p * c * dt_hours for p, c in zip(results['prices'], results['charge']))
        results['degradation_cost'] = (results['total_charged'] + results['total_discharged']) * self.degradation_cost
        results['net_profit'] = results['revenue'] - results['cost'] - results['degradation_cost']
        
        # Battery health metrics
        results['equivalent_cycles'] = results['total_discharged'] / self.battery_capacity
        results['remaining_cycle_life'] = self.cycle_life - results['equivalent_cycles']
        
        return results

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Enhanced visualization of optimization results."""
        try:
            results = self.get_results()
            if not results:
                logger.warning("No results to plot")
                return

            # Create figure and subplots
            plt.style.use('default')
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(3, 1, height_ratios=[2, 1.5, 1])

            # Plot 1: Prices and Power
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(results['datetime'], results['prices'], 'k-', label='Electricity Price', linewidth=2)
            ax1.set_ylabel('Price ($/kWh)')
            ax1.grid(True, alpha=0.3)
            
            ax1_twin = ax1.twinx()
            ax1_twin.fill_between(results['datetime'], results['charge'], 
                                color='g', alpha=0.3, label='Charging')
            ax1_twin.fill_between(results['datetime'], [-d for d in results['discharge']], 
                                color='r', alpha=0.3, label='Discharging')
            ax1_twin.set_ylabel('Power (kW)')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Plot 2: State of Charge
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(results['datetime'], results['soc'], 'b-', label='State of Charge', linewidth=2)
            ax2.fill_between(results['datetime'], self.min_soc, self.max_soc, 
                            color='gray', alpha=0.2, label='Operating Range')
            ax2.axhline(y=self.max_soc, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=self.min_soc, color='r', linestyle='--', alpha=0.5)
            ax2.set_ylabel('State of Charge')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Plot 3: Cumulative Revenue
            ax3 = fig.add_subplot(gs[2])
            cumulative_profit = np.cumsum([
                p * d * self.time_resolution.total_seconds() / 3600 - 
                p * c * self.time_resolution.total_seconds() / 3600
                for p, d, c in zip(results['prices'], results['discharge'], results['charge'])
            ])
            ax3.plot(results['datetime'], cumulative_profit, 'g-', label='Cumulative Profit')
            ax3.set_ylabel('Cumulative Profit ($)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # Format x-axis
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
                if ax != ax3:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time of Day')

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
                
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error in plotting: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

def main():
    """Run continuous optimization with realistic parameters."""
    try:
        # Initialize optimizer with realistic parameters
        optimizer = EnergyArbitrageOptimizer(
            battery_capacity_kwh=100,
            max_power_kw=20,
            battery_efficiency=0.92,
            initial_soc=0.3,
            min_soc=0.1,
            max_soc=0.9,
            cycle_life=3000,
            calendar_life_years=10,
            battery_cost=400,
            time_resolution='15min'
        )

        # Run optimization
        if optimizer.optimize():
            results = optimizer.get_results()
            
            logger.info("\nOptimization Results:")
            logger.info(f"Total Revenue: ${results['revenue']:.2f}")
            logger.info(f"Total Cost: ${results['cost']:.2f}")
            logger.info(f"Degradation Cost: ${results['degradation_cost']:.2f}")
            logger.info(f"Net Profit: ${results['net_profit']:.2f}")
            logger.info(f"Energy Charged: {results['total_charged']:.1f} kWh")
            logger.info(f"Energy Discharged: {results['total_discharged']:.1f} kWh")
            logger.info(f"Round-trip Efficiency: {results['round_trip_efficiency']*100:.1f}%")
            logger.info(f"Equivalent Cycles: {results['equivalent_cycles']:.2f}")
            
            # Plot and save results
            logger.info("Generating plots...")
            optimizer.plot_results(save_path='optimization_results.png')
            logger.info("Plotting complete")
        else:
            logger.error("Optimization failed")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()