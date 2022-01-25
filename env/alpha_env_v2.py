
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

    def __init__(self, param, isTrain):
        self.isTrain = isTrain
        if self.isTrain == True:
            df = pd.read_csv(param['DTRAIN_PATH'])
        else:
            df = pd.read_csv(param['DTEST_PATH'])


        self.stock_all = df['thscode'].unique() # len=454
        self.stock_list = df['thscode']

        self.profit = 0
        self.flow = 0
        self.portfolio_list = []    # portfolio at every step
        self.buyhold_list = []      # buy_hold at every step

        self.data_train = df.drop(['CLOSE_AFTER'], axis=1)
        self.close_train = df['CLOSE_AFTER']
        self.time_stump = df['time']

        self.K = param['HOLDING_PERIOD']
        self.test_stock_num = param['TEST_NUM_STOCK']

        self.all_stock_close = []   # (stock_num, time_step)
        self.all_stock_feature = [] # (stock_num, time_step, feature, dim)
        for i in range(self.test_stock_num):
            thscode = self.stock_all[i]
            dt = self.data_train[self.stock_list == thscode]
            stock_i_feature = np.array(dt.iloc[:, 3:])      # TODO: pay attention to the feature dimension
            self.all_stock_feature.append(stock_i_feature)
            self.all_stock_close.append(self.close_train[self.stock_list == thscode])

        self.all_stock_close = np.array(self.all_stock_close).transpose((1,0)) # (time_step, stock_num)
        self.all_stock_feature = np.array(self.all_stock_feature).transpose((1,0,2)) # (time_step, stock_num, feature_dim)

        if self.isTrain == True:
            self.seq_time = param['SEQ_TIME']
        else:
            self.seq_time = len(self.all_stock_close) - self.K

    def reset(self):
        self.inventory = 0
        self.initial_money = 10000
        self.total_money = 10000
        self.profit = 0
        self.profit_list = []       # profit at every timestump
        self.portfolio_list = []    # portfolio at every step
        self.buyhold_list = []      # buy_hold at every step
        self.stock_price = 0
        self.today_buy_port = 0 # constraint for daily trading
        # measurement
        self.buy_hold = 0
        self.sp = 0
        self.maxdrawdown = 0
        self.mdd = 0
        self.romad = 0

        Portfolio_unit = 1
        Rest_unit = 1

        self.holding = [0] * self.test_stock_num
        self.t = 0
        self.time_stump1 = self.time_stump[self.stock_list == self.stock_all[0]]

        if self.isTrain == True:
            self.trade_date = np.random.randint(self.K - 1, len(self.time_stump1) - self.seq_time)  # randomly pick a start point for trading
        else:
            self.trade_date = self.K - 1
        # feature for all stock at trade_date+t and previous K step
        # all_stock_feature_in_K: (window_size_K, stock_num, feature_dim)
        all_stock_feature_in_K = self.all_stock_feature[self.trade_date+self.t-self.K+1 : self.trade_date+self.t+1]
        # all_stock_his_state: (num_stock, window_size_k, feature_dim)
        all_stock_his_state = all_stock_feature_in_K.transpose((1,0,2))

        self.init_price = self.all_stock_close[self.trade_date]  # price vector (all stock) at self.trade_date
        equal_invest = self.initial_money / self.test_stock_num
        self.equal_holding = list(map(lambda x: (equal_invest / x[0]) * x[1], zip(self.init_price, [1]*self.test_stock_num)))
        return all_stock_his_state

    def get_K_his_state(self, feature, time_stamp):
        # K history state for stock i at time t
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
        # action: vector \in {0,1,-1}

        self.stock_price = self.all_stock_close[self.trade_date + self.t] # price vector (all stock) at self.trade_date+t

        ave_price = np.mean(self.stock_price)
        #action = list(map(lambda x: x[0]*x[1]/ave_price, zip(self.stock_price, action)))
        action = list(map(lambda x: (100/x[0]) * x[1] , zip(self.stock_price, action)))

        # holding>=0
        action0 = action
        judge_trade = list(map(lambda x: x[0]+x[1], zip(self.holding, action0)))    # judge whether a valid trade
        for i in range(len(action0)):
            if judge_trade[i]>=0:
                action[i] = action0[i]
            else:
                action[i]=0
        self.holding = list(map(lambda x: x[0]+x[1], zip(self.holding, action)))
        self.cost = sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, action)))) # +: spending，-: earning

        # holding_check = [int(i) for i in self.holding]
        # print(holding_check)

        self.total_money -= self.cost

        self.Portfolio_unit = (self.total_money + sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, self.holding))))
                               ) / self.initial_money
        self.buy_hold = sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, self.equal_holding)))) / self.initial_money

        Rest_unit = self.total_money / self.initial_money

        total_profit = (self.total_money + sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, self.holding))))
                        ) - self.initial_money

        self.profit = total_profit / self.initial_money

        self.profit_list.append(self.profit)
        self.portfolio_list.append(self.Portfolio_unit)
        self.buyhold_list.append(self.buy_hold)

        self.t += 1

        done = self.seq_time < (self.t + 1)

        sp_std = np.std(self.profit_list)
        if sp_std<10e-4:
            sp_std=10e-4
        self.sp = (np.mean(self.profit_list))/sp_std

        reward = self.get_reward(self.sp)

        # all_stock_feature_in_K: (window_size_K, stock_num, feature_dim)
        all_stock_feature_in_K = self.all_stock_feature[self.trade_date+self.t-self.K+1 : self.trade_date+self.t+1]
        # all_stock_his_state: (num_stock, window_size_k, feature_dim)
        all_stock_his_state = all_stock_feature_in_K.transpose((1,0,2))
        state = all_stock_his_state

        return state, reward, done, {}
