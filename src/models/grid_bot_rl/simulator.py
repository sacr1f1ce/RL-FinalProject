import numpy as np
from market import StockMarket
from typing import Tuple
import pandas as pd


class Simulator:

    def __init__(
        self,
        system: StockMarket,
        market_data: pd.DataFrame,  # price + features
    ):
        self.system = system
        self.N_steps = N_steps
        self.state = np.copy(state_init)
        self.state_init = np.copy(state_init)
        self.current_step_idx = 0

    def step(self) -> bool:
        """Do one step

        Returns:
            bool: status of simulation. `True` - simulation continues, `False` - simulation stopped
        """

        if self.current_step_idx <= self.N_steps:
            current_market_state = self.market_data.iloc[self.current_step_idx]
            self.state = (
                current_market_state.Open,
                current_market_state.High,
                current_market_state.Low,
                current_market_state.Close,
                current_market_state.Volume,
                current_market_state.macd,
                current_market_state.ema26,
                current_market_state.ema12,
                current_market_state.force_index
            )

            self.current_step_idx += 1
            return True
        else:
            return False

    def reset(self) -> None:
        """Resets the system to initial state"""

        self.state = np.copy(self.state_init)
        self.current_step_idx = 0
        self.system.reset()

    def get_sim_step_data(self) -> Tuple[np.array, np.array, int]:
        """Get current observation, action and step id

        Returns:
            Tuple[np.array, np.array, int]:
        """

        return (
            self.system.get_observation(self.state),
            np.copy(self.system.action),
            int(self.current_step_idx),
        )
