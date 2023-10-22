import numpy as np


class StockMarket:
    """System class: stock market"""

    dim_action: int = 1
    dim_observation: int = 4
    dim_state: int = 4

    def __init__(self) -> None:
        """Initialize `InvertedPendulumSystem`"""

        self.reset()

    def reset(self) -> None:
        """Reset system to inital state."""

        self.action = np.zeros(self.dim_action)

    def receive_action(self, action: np.array) -> None:
        """Save current action to `self.action`

        Args:
            action (np.array): current action
        """

        self.action = action

    @staticmethod
    def get_observation(state: np.array) -> np.array:
        """Get observation given a state

        Args:
            state (np.array): system state

        Returns:
            np.array: observation
        """
        observation = state

        return observation
