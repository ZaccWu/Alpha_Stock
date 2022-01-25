import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
from collections import deque
from env.alphatest_env import ALPHA_ENV

# Hyper Parameters
MAX_EPISODE = 15000  # Episode limitation
MAX_STEP = 1000  # Step limitation in an episode
TEST = 5  # The number of experiment test every 100 episode
TEST_EPI = 100 # How often we test
SAVE_FREQ = 1000

# Hyper Parameters for PG Network
GAMMA = 0.95  # discount factor
LR = 0.01  # learning rate
LSTM_H_DIM = 128
ATTN_W_DIM = 64
LSTM_DROPOUT = 0.2
LSTM_LAYER = 1
CANN_HEADS = 4
STOCK_POOL = 7  # how many candidate stocks (G in the paper)

# Parameters for environment
param = {
    'SEQ_TIME': 48,
    'HOLDING_PERIOD': 12,       # look back history (K in the paper)
    'TEST_NUM_STOCK': 25,       # 选取多少支股票来�?    #'DT_PATH': 'zztestn.csv',   # 数据文件路径
    # 'DT_PATH': 'combine/2018_zz500.csv',
    'DT_PATH': 'zztestn.csv',
}

# Parameters depend on your data
FEATURE_DIM = 29    # 股票特征多少�?
# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMHA(nn.Module):
    def __init__(self, lstm_input_size=FEATURE_DIM, lstm_h_dim=LSTM_H_DIM,
                 lstm_num_layers=LSTM_LAYER,
                 attn_w_dim=ATTN_W_DIM, dropout=LSTM_DROPOUT):
        """
        接收窗口大小为K的固定长度序列作为输�?        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_h_dim,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout)
        self.ll = nn.Linear(param['HOLDING_PERIOD'], 1)

    def forward(self, x):
        """
        x的形状：batch_size * stock_num * window_size_K * feature_dim (ndim=4)
        """
        # if is aligned batch inputs (B * I * T * E_feat), ndim=4
        batch_size = x.shape[0]
        num_stock = x.shape[1]

        # lstm设置为batch_first则需要输�?batch, seq_len, input_size)的数�?        # 在这里让x变为(batch_size * stock_num, window_size_K, feature_dim)的张�?
        x = x.view(batch_size * num_stock, x.shape[-2], x.shape[-1])

        # 设置了batch_fisrt=True后，单向lstm输出outputs(batch, seq_len, hidden_size), (hn, cn)

        outputs, _ = self.lstm(x)           # (batch*stock_num, window_size_K, hidden_size)
        outputs = outputs.transpose(1,2)    # (batch*stock_num, hidden_size, window_size_K)

        ll_input = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1) # (batch_size * stock_num * hidden_size, window_size_K)
        ll_output = self.ll(ll_input)
        # ll_output: (batch_size * num_stock * hidden_size, 1)

        out_rep = ll_output.reshape(outputs.shape[0], outputs.shape[1]) # (batch_size * num_stock, hidden_size, )
        out_rep = out_rep.view(batch_size, num_stock, -1)   # (batch_size, stock_num, hidden_size)
        return out_rep


class PGNetwork(nn.Module):
    def __init__(self):
        super(PGNetwork, self).__init__()
        self.lstm_ha = LSTMHA()
        self.ll = nn.Linear(LSTM_H_DIM, 1)   # 测试中将每只股票Embedding转化�?int的分�?

    def forward(self, x):
        if x.dim() == 3:
            x_b = x.unsqueeze(0)  # TODO: 第一个维度给batch
        else:
            x_b = x

        # x_b: (batch_size, num_stock, window_size_K, feature_dim)
        ha_rep = self.lstm_ha(x_b)
        # ha_rep: (batch_size, num_stock, E_c)
        ll_input = ha_rep.reshape(ha_rep.shape[0]*ha_rep.shape[1],-1) # here batch_size=1
        # ll_input: (batch_size * num_stock, E_c)
        ll_output = self.ll(ll_input)
        # ll_output: (batch_size * num_stock, 1)
        stock_score = ll_output.reshape(ha_rep.shape[0], ha_rep.shape[1])
        # stock_score: (batch_size, num_stock, 1)

        stock_score = stock_score.squeeze(-1)
        sorted_x, sorted_indices = torch.sort(stock_score, dim=-1, descending=True)

        winner_assets_x = sorted_x[:, :STOCK_POOL]
        # winner_assets_indices = sorted_indices[:, :self.G]
        loser_assets_x = sorted_x[:, -STOCK_POOL:]
        # loser_assets_indices = sorted_indices[:, :self.G]

        # --- Todo: temp: the following codes suit only bsz = 1 -----#
        winner_proportion = F.softmax(winner_assets_x, dim=-1)
        loser_proportion = -F.softmax(1 - loser_assets_x, dim=-1)

        zeros_assets = torch.zeros_like(sorted_x[:, STOCK_POOL:-STOCK_POOL],
                                        requires_grad=True).type(x.dtype)
        # sorted_winner_assets_x = winner_assets_x.sort_values(by=sorted_indices)
        # sorted_loser_assets_x = loser_assets_x.sort_values(by=sorted_indices)

        b_c = torch.cat([winner_proportion, zeros_assets,
                         loser_proportion], dim=1)


        return b_c, sorted_indices, stock_score

class PG(object):
    # dqn Agent
    def __init__(self, env):  # 初始�?        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        # self.network = PGNetwork().to(device)
        self.network = torch.load('modelsave/lstm_net_3000.pkl')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        # TODO: 解决输入state维度问题
        observation = torch.FloatTensor(observation).to(device)
        network_output, sorted_indices, factor = self.network.forward(observation)
        action_prob = network_output.detach().cpu().numpy()[0]

        # action_prob的形�? [0.4, 0.1, 0,..., 0.2, 0.3]
        # action_prob=np.array([0.4,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.3])

        action_shuffle = [np.random.binomial(1,i)
                          if i>=0 else -np.random.binomial(1,-i)
                          for i in action_prob
                          ] # 按概率决定是否交�?        # action_shuffle的形�? [1, 0, 0,..., -1, 0]（随机向量）

        action = [0]*len(action_shuffle)
        sorted_indices = sorted_indices.detach().cpu().numpy()[0]

        for i in range(len(action_shuffle)):
            action[sorted_indices[i]] = action_shuffle[i]

        return action, factor

def main():
    env = ALPHA_ENV(param)
    agent = PG(env)
    total_reward = 0
    factor_list = []

    state = env.reset()
    for j in range(MAX_STEP):
        action, factor = agent.choose_action(state)  # direct action for test
        state, reward, done, _ = env.step(action)
        total_reward += reward
        factor_list.append(factor.detach().cpu().numpy()[0])
        if done:
            break

    factor_list = pd.DataFrame(factor_list)
    factor_list.to_csv('factor_lstm.csv',index=False)
    print(total_reward)

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
