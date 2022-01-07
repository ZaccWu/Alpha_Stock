
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from collections import deque
import random
import time

import gym
import os
import h5py


class ALPHA_ENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        #df = pd.read_csv(r'local_zz500b.csv')
        #df = pd.read_csv(r'combine/2018_zz500.csv')
        df = pd.read_csv(r'zztestn.csv')
        self.today = '2018/1/2'

        self.stock_all = df['thscode'].unique() # len=454
        self.stock_list = df['thscode']

        self.action_space = gym.spaces.Box(
            low=np.array([-1] * 1),
            high=np.array([1] * 1),
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-5] * 31),
            high=np.array([5] * 31),
        )

        self.seq_time = 20
        self.profit = 0
        self.flow = 0

        self.data_train = df.drop(['CLOSE_AFTER'], axis=1)
        self.close_train = df['CLOSE_AFTER']
        self.time_stump = df['time']

        # TODO: 临时参数
        self.K = 12 # 历史期为15
        self.test_stock_num = 25    # 先用25支股票测试

        self.all_stock_close = []   # (stock_num, time_step)
        self.all_stock_feature = [] # (stock_num, time_step, feature,dim)
        for i in range(self.test_stock_num):
            thscode = self.stock_all[i]
            dt = self.data_train[self.stock_list == thscode]
            stock_i_feature = np.array(dt.iloc[:, 3:])
            self.all_stock_feature.append(stock_i_feature)
            self.all_stock_close.append(self.close_train[self.stock_list == thscode])

        self.all_stock_close = np.array(self.all_stock_close).transpose((1,0)) # (time_step, stock_num)
        self.all_stock_feature = np.array(self.all_stock_feature).transpose((1,0,2)) # (time_step, stock_num, feature_dim)

    def reset(self):
        self.inventory = 0
        self.initial_money = 1000000
        self.total_money = 1000000
        self.profit = 0
        self.profit_list = []       # 每个时间点的profit
        self.portfolio_list = []    # 每个step的资产
        self.stock_price = 0
        self.today_buy_port = 0 # 限制每日交易的
        # 测量指标
        self.buy_hold = 0
        self.sp = 0
        self.maxdrawdown = 0
        self.mdd = 0
        self.romad = 0


        Portfolio_unit = 1
        Rest_unit = 1

        self.holding = [0] * self.test_stock_num
        self.t = 0


        self.time_stump1 = self.time_stump[self.stock_list == self.stock_all[0]]  # 这支股票的时间点
        self.trade_date = np.random.randint(self.K - 1, len(self.time_stump1) - self.seq_time)  # 随机选交易开始时间点


        # 所有股票在时间点trade_date+t时到往前K步的特征
        # all_stock_feature_in_K: (window_size_K, stock_num, feature_dim)
        all_stock_feature_in_K = self.all_stock_feature[self.trade_date+self.t-self.K+1 : self.trade_date+self.t+1]
        # 返回列表的维数: (num_stock, window_size_k, feature_dim)
        all_stock_his_state = all_stock_feature_in_K.transpose((1,0,2))

        return all_stock_his_state

    def get_K_his_state(self, feature, time_stamp):
        # 第i支股票在t时间点往前K步的历史记录
        k_his_state = [feature[time_stamp - self.K + j] for j in range(self.K)]  # (window_size_K, feature_dim)
        #print('feature len:', len(feature))
        return k_his_state


    def get_reward(self, profit):
        reward = 0
        if 0 < profit <= 0.1:
            reward = 1
        if 0.1 < profit <= 0.2:
            reward = 2
        if 0.2 <= profit:
            reward = 4
        if -0.1 <= profit < 0:
            reward = -1
        if -0.2 <= profit < -0.1:
            reward = -2
        if profit < -0.2:
            reward = -4
        return reward

    def step(self, action):
        # TODO: 解决action不同的问题
        # 传入的action：是否买卖股票的向量（按index顺序）\in {0,1,-1}

        self.stock_price = self.all_stock_close[self.trade_date + self.t] # 所有股票这个时间的价格向量
        ave_price = np.mean(self.stock_price)
        #action = list(map(lambda x: x[0]*x[1]/ave_price, zip(self.stock_price, action))) # 价格normalize
        action = list(map(lambda x: x[0] * x[1] , zip(self.stock_price, action)))

        # 所有股票的time_stump一样的，就默认用最后一支了
        today_time = (self.time_stump1.iloc[self.trade_date + self.t]).split(' ')[0]  # 获取日期到天

        # 无买卖限制
        # self.holding = list(map(lambda x: x[0]+x[1], zip(self.holding, action)))
        # self.cost = sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, action)))) # +表示支出，-表示收入

        # 禁止做空
        action0 = action
        judge_trade = list(map(lambda x: x[0]+x[1], zip(self.holding, action0)))    # 判断是否可以交易
        for i in range(len(action0)):
            if judge_trade[i]>=0:
                action[i] = action0[i]
            else:
                action[i]=0
        self.holding = list(map(lambda x: x[0]+x[1], zip(self.holding, action)))
        self.cost = sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, action)))) # +表示支出，-表示收入

        # holding_check = [int(i) for i in self.holding]
        # print(holding_check)

        self.total_money -= self.cost

        self.Portfolio_unit = (self.total_money + sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, self.holding))))
                               ) / self.initial_money  # 资产与初始资金比例
        Rest_unit = self.total_money / self.initial_money                           # 剩余金额占比

        total_profit = (self.total_money + sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, self.holding))))
                        ) - self.initial_money

        self.profit = total_profit / self.initial_money

        self.profit_list.append(self.profit)
        self.portfolio_list.append(self.Portfolio_unit)

        self.t += 1
        done = self.seq_time < (self.t + 1)

        # if done:
        #     check_holding = [int(i) for i in self.holding]
        #     print(check_holding)

        sp_std = np.std(self.profit_list)
        if sp_std<10e-4:
            sp_std=10e-4
        self.sp = (np.mean(self.profit_list))/sp_std          # 最后输出全时间段的夏普率（无风险利率3%）

        reward = self.get_reward(self.sp)

        # 所有股票在时间点trade_date+t时到往前K步的特征
        # all_stock_feature_in_K: (window_size_K, stock_num, feature_dim)
        all_stock_feature_in_K = self.all_stock_feature[self.trade_date+self.t-self.K+1 : self.trade_date+self.t+1]
        # 返回列表的维数: (num_stock, window_size_k, feature_dim)
        all_stock_his_state = all_stock_feature_in_K.transpose((1,0,2))
        state = all_stock_his_state

        return state, reward, done, {}
