from scipy.linalg import norm
import numpy as np
from tqdm import tqdm


class AdaptiveGridBot:
    def __init__(
            self,
            market_data,
            update_step=10, # in minutes
            levels_num=10,
            levels_step=0.001,  # initial level step
            balance=100000,
            fee=0.2181 / 100,
            above_level_pct=0.2,
            use_up=None
    ):
        assert 0 < levels_step < 1, f'levels_step must be in (0, 1) range, {levels_step}'
        assert levels_num >= 1, f'levels_num must be greater than 0, {levels_num}'

        self.data = market_data
        self.update_step = update_step
        self.levels_num = levels_num
        self.levels_step = levels_step
        self.balance = balance
        self.history = []
        self.fee = fee
        self.buy_points = []
        self.sell_points = []
        self.use_up = use_up
        self.above_level_pct = above_level_pct

        self.current_step_idx = 0

        self.levels_step_init = levels_step
        self.balance_init = balance

        self.assets = 0
        self.bought = [False for _ in range(self.levels_num)]
        self.action = 0

    def get_buy_levels(self):
        self.buy_amount = self.balance / self.levels_num
        self.buy_orders = [
            self.ref_price * (1-k * self.levels_step) for k in range(1, self.levels_num+1)
        ]

    def get_sell_levels(self):
        if isinstance(self.use_up, bool):
            if self.use_up:
                self.sell_orders = [self.ref_price * (1 + k * self.levels_step) for k in range(self.levels_num)]
            else:
                self.sell_orders = [self.ref_price * (1 - k * self.levels_step) for k in range(self.levels_num)]

        else:
            self.sell_orders = []
            for k in range(self.levels_num):
                if k > int(self.levels_num*self.above_level_pct):
                    k = k - int(self.levels_num*self.above_level_pct)
                    self.sell_orders += [self.ref_price * (1 - k * self.levels_step)]
                else:
                    self.sell_orders += [self.ref_price * (1 + k * self.levels_step)]

    def step(self):
        if self.action > 0:
            self.levels_step = self.action
        self.ref_price = self.data.iloc[self.update_step * self.current_step_idx].Open
        self.get_buy_levels()
        self.get_sell_levels()

        for index, row in self.data.iloc[
            self.update_step * self.current_step_idx:self.update_step * (self.current_step_idx + 1)
        ].iterrows():

            sell_amount = self.assets / sum(self.bought) if sum(self.bought) else 0

            for i in range(len(self.bought)):
                if self.bought[i] and row.High >= self.sell_orders[i]:  # selling
                    self.bought[i] = False

                    self.balance += (
                        sell_amount * self.sell_orders[i] * (1 - self.fee)
                    )
                    self.assets -= sell_amount

                    self.history += [self.balance]
                    self.sell_points += [(index, self.sell_orders[i])]

                elif not self.bought[i] and row.Low <= self.buy_orders[i] and self.balance >= self.buy_amount:  # buying
                    self.bought[i] = True
                    self.balance -= self.buy_amount
                    self.assets += (1 - self.fee) * self.buy_amount / self.buy_orders[i]
                    self.history += [self.balance]
                    self.buy_points += [(index, self.buy_orders[i])]

        self.current_step_idx += 1
        return self.current_step_idx * (self.update_step + 1) < len(self.data)

    def get_sim_step_data(self):
        last_row = self.data.iloc[self.update_step * (self.current_step_idx + 1) - 1]
        observation = (
            last_row.ema12,
            last_row.ema26,
            last_row.macd,
            last_row.force_index,
            float(self.balance + self.assets * last_row.Close)
        )
        return np.array(observation), np.copy(self.action), int(self.current_step_idx)

    def receive_action(self, action):
        self.action = action

    def reset(self):
        self.levels_step = self.levels_step_init
        self.balance = self.balance_init
        self.history = []
        self.buy_points = []
        self.sell_points = []
        self.current_step_idx = 0
        self.assets = 0
        self.bought = [False for _ in range(self.levels_num)]

    # STEPS_UPDATE,
    #STATE INIT