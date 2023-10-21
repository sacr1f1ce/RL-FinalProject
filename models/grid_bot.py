from tqdm import tqdm
import pandas as pd


class GridBot:
    def __init__(self, levels_num=2, levels_step=0.001):
        assert 0 < levels_step < 1, f'levels_step must be in (0, 1) range, {levels_step}'
        assert levels_num >= 1, f'levels_num must be greater than 0, {levels_num}'

        self.levels_num = levels_num
        self.levels_step = levels_step
        self.balance = 0
        self.history = []
        self.bought = [False for _ in range(self.levels_num)]

    def trade(self, df_path):
        df = pd.read_csv(df_path)
        # use_mean - use mean price over last month as a ref level, else use first open price
        ref_price = df.iloc[0]['Open']
        buy_orders = [ref_price * (1 - k * self.levels_step) for k in range(1, self.levels_num + 1)]
        sell_orders = [ref_price * (1 - k * self.levels_step) for k in range(self.levels_num)]
        for index, row in tqdm(df.iterrows()):
            for i in range(len(self.bought)):
                if self.bought[i] and row.High >= sell_orders[i]:
                    self.bought[i] = False
                    self.balance += sell_orders[i]
                    self.history += [self.balance]

                elif not self.bought[i] and row.Low <= buy_orders[i]:
                    self.bought[i] = True
                    self.balance -= buy_orders[i]
                    self.history += [self.balance]
        

grid_bot = GridBot()
grid_bot.trade('data/raw/btcusdt.csv')
print(grid_bot.balance)
print(grid_bot.history)
