from scipy.linalg import norm
import numpy as np
from tqdm import tqdm


class AdaptiveGridBot:
    def __init__(
            self,
            policy,
            running_reward,
            discount,  # 0.99
            levels_num=4,
            levels_step=0.01,  # initial level step
            balance=100000,
            fee=0.2181 / 100,
            use_up=False
    ):
        assert 0 < levels_step < 1, f'levels_step must be in (0, 1) range, {levels_step}'
        assert levels_num >= 1, f'levels_num must be greater than 0, {levels_num}'

        self.__policy = policy
        self.__running_reward = running_reward
        self.__discount = discount

        self.levels_num = levels_num
        self.levels_step = levels_step
        self.balance = balance
        self.history = []
        self.fee = fee
        self.buy_points = []
        self.sell_points = []
        self.use_up = use_up

    def get_buy_levels(self):
        self.buy_amount = self.balance / self.levels_num
        self.buy_orders = [
            self.ref_price * (1-k * self.levels_step) for k in range(1, self.levels_num+1)
        ]

    def get_sell_levels(self):
        if self.use_up:
            self.sell_orders = [
                self.ref_price * (1+k * self.levels_step) for k in range(self.levels_num)
            ]
        else:
            self.sell_orders = [
                self.ref_price * (1-k * self.levels_step) for k in range(self.levels_num)
            ]

    def trade(self, market_data, update_step=10):  # update_step in minutes
        self.assets = 0
        self.bought = [False for _ in range(self.levels_num)]
        self.get_buy_levels()
        self.get_sell_levels()

        cnt = 0
        for index, row in tqdm(market_data.iterrows()):

            sell_amount = self.assets / sum(self.bought) if sum(self.bought) else 0

            for i in range(len(self.bought)):

                if cnt == update_step:  # updating the grid step
                    cnt = 0
                    self.levels_step = self.__policy(row)
                    self.get_buy_levels()
                    self.get_sell_levels()
                    
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

                cnt += 1
        self.balance += self.assets * market_data.iloc[-1].Close * (1 - self.fee)  # selling the rest if we have any


        trajectory = [self.__initial_state]
        actions = []
        total_reward = 0
        accumulated_discount = 1
        for _ in range(steps):
            current_state = trajectory[-1]
            control_input = self.__controller(parameters, current_state)
            actions.append(control_input)
        
        next_state = self.__state_transition_function(current_state, control_input)
        total_reward += self.__running_reward(current_state, control_input) * accumulated_discount
        accumulated_discount *= self.__discount

        trajectory.append(next_state)
        #return np.array(trajectory), np.array(actions), total_reward
    


slingshot_initial_state = np.array([-1.0, 0.0, 0.0, 0.0])




def step(parameters, sample_size=20, learning_rate=0.1):
    average_shift = 0
    average_total_reward = 0
    for _ in range(sample_size):
        _, actions, total_reward = slingshot_system.run_with_parameters(parameters)
        average_shift += (actions[0] - parameters) * total_reward / sample_size
        average_total_reward += total_reward / sample_size
    return project(parameters + learning_rate * average_shift), average_total_reward, norm(learning_rate * average_shift) / 0.04


