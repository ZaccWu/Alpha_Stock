
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
        df = pd.read_csv(r'zztest.csv')
        #df = pd.read_csv(r'combine/2018_zz500.csv')
        self.today = '2018/1/2'


        self.stock_all = df['thscode'].unique() # len=454
        self.test_count = 0  # for testing
        self.stock_list = df['thscode']
        self.close = df['CLOSE_AFTER']

        self.action_space = gym.spaces.Box(
            low=np.array([-1] * 1),
            high=np.array([1] * 1),
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-5] * 31),
            high=np.array([5] * 31)
        )

        self.seq_time = 48
        self.profit = 0
        self.flow = 0

        self.data_train = df.drop(['CLOSE_AFTER'], axis=1)
        self.close_train = df['CLOSE_AFTER']
        self.time_stump = df['time']

        # TODO: 临时参数
        self.K = 10 # 历史期为10
        self.test_stock_num = 30    # 先用30支股票测试


    def reset(self):
        # if self.util == 'test':
        #     thscode = self.stock_all[self.test_count]  # 测试时使用指定测试股票
        #     self.dt = self.data_train[self.stock_list == thscode]
        #     self.dt1 = np.array(self.dt.iloc[:, 3:])
        #     self.close1 = self.close_train[self.stock_list == thscode]
        #     self.time_stump1 = self.time_stump[self.stock_list == thscode]
        #
        #     self.test_count+=1
        #     self.trade_date = 0 # 测试时从头开始跑
        #else:
        # 场景的设置
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

        #thscode = random.choice(self.stock_all)   # 训练时随机选择股票
        # self.dt = self.data_train[self.stock_list == thscode]
        # self.dt1 = np.array(self.dt.iloc[:, 3:])
        # self.close1 = self.close_train[self.stock_list == thscode]
        # self.time_stump1 = self.time_stump[self.stock_list == thscode]
        # self.trade_date = np.random.randint(self.K-1, len(self.close1) - self.seq_time)    # 训练时随机选择时间点开始


        Portfolio_unit = 1
        Rest_unit = 1

        self.holding = [0] * self.test_stock_num

        self.t = 0

        # 所有股票再t时间点往前K步的历史记录
        all_stock_his_state = []

        self.time_stump1 = self.time_stump[self.stock_list == self.stock_all[0]]  # 这支股票的时间点
        self.trade_date = np.random.randint(self.K - 1, len(self.time_stump1) - self.seq_time)  # 随机选交易开始时间点

        self.all_stock_close = []

        for i in range(self.test_stock_num):
            thscode = self.stock_all[i]
            self.dt = self.data_train[self.stock_list == thscode]
            #print("dt len:",len(self.dt))
            stock_i_feature = np.array(self.dt.iloc[:, 3:])    # 这支股票的特征（所有时间点）

            self.all_stock_close.append(self.close_train[self.stock_list == thscode])  # 这支股票的收盘价
            #print("trade time:", self.trade_date+self.t)
            # 股票i在时间点trade_date+t时到往前K步的特征
            k_his_state = self.get_K_his_state(stock_i_feature, self.trade_date+self.t)

            # 所有股票K时间窗的特征
            all_stock_his_state.append(k_his_state)

        # check the dimension
        # print(len(self.all_stock_close),len(self.all_stock_close[0]))
        closeT = np.array(self.all_stock_close)
        self.all_stock_close = closeT.transpose((1,0))
        # 维数：(all_time, num_stock)
        # print(len(self.all_stock_close),len(self.all_stock_close[0]))

        # 返回列表的维数: (num_stock, window_size_k, feature_dim)
        return all_stock_his_state

    def get_K_his_state(self, feature, time_stamp):
        # 第i支股票在t时间点往前K步的历史记录
        k_his_state = []  # (window_size_K, feature_dim)
        #print('feature len:', len(feature))
        for j in range(self.K):
            his_state = feature[time_stamp - self.K + j]
            # add_state = np.array([Portfolio_unit, Rest_unit]).flatten()
            # his_state = np.hstack([his_state, add_state])
            k_his_state.append(his_state)
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

        # self.all_stock_close[i]是第i支股票所有时段的价格
        #self.stock_price = self.close1.iloc[self.trade_date + self.t]

        self.stock_price = self.all_stock_close[self.trade_date + self.t] # 所有股票这个时间的价格向量

        # 所有股票的time_stump一样的，就默认用最后一支了
        today_time = (self.time_stump1.iloc[self.trade_date + self.t]).split(' ')[0]  # 获取日期到天

        # TODO：先复现最简单的情况：无买卖限制
        self.holding = list(map(lambda x: x[0]+x[1], zip(self.holding, action)))
        self.cost = sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, action)))) # +表示支出，-表示收入
        self.total_money -= self.cost

        self.Portfolio_unit = (self.total_money + sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, self.holding))))
                               ) / self.initial_money  # 资产与初始资金比例
        Rest_unit = self.total_money / self.initial_money                           # 剩余金额占比

        total_profit = (self.total_money + sum(list(map(lambda x: x[0]*x[1], zip(self.stock_price, self.holding))))
                        ) - self.initial_money
        # reward = self.get_reward(total_profit / self.initial_money)                 # 传入get_reward的就是收益率
        self.profit = total_profit / self.initial_money # get profit（收益率）

        self.profit_list.append(self.profit)
        self.portfolio_list.append(self.Portfolio_unit)

        self.t += 1
        done = self.seq_time < (self.t + 1)

        sp_std = np.std(self.profit_list)
        if sp_std<10e-4:
            sp_std=10e-4
        self.sp = (np.mean(self.profit_list))/sp_std          # 最后输出全时间段的夏普率（无风险利率3%）

        reward = self.get_reward(self.sp)

        # print(reward, self.holding)

        # 所有股票再t时间点往前K步的历史记录
        all_stock_his_state = []
        for i in range(self.test_stock_num):
            thscode = self.stock_all[i]
            self.dt = self.data_train[self.stock_list == thscode]
            stock_i_feature = np.array(self.dt.iloc[:, 3:])    # 这支股票的特征
            k_his_state = self.get_K_his_state(stock_i_feature, self.trade_date+self.t)
            all_stock_his_state.append(k_his_state)

        state = all_stock_his_state


        return state, reward, done, {}
