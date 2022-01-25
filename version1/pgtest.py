import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
from collections import deque
from env.alpha_env import ALPHA_ENV

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
    #'DT_PATH': 'combine/2018_zz500.csv',
    'DT_PATH': 'zztestn.csv',
}

# Parameters depend on your data
FEATURE_DIM = 29    # 股票特征多少�?
# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StdAttn(nn.Module):
    def __init__(self, input_size, att_dim, dropout):
        super().__init__()
        self.attention_dim = att_dim
        self.query_transform = nn.Linear(input_size, att_dim)
        self.key_transform = nn.Linear(input_size, att_dim)
        self.attn_score = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(att_dim, 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, q, k, v):
        # q, v都是(batch*stock_num, window_size_K, hidden_size)
        # k�?batch*stock_num, hidden_size)
        embed_dim = q.shape[-1] # hidden_size
        bsz = q.shape[0]        # batch*stock_num
        num_queries = q.shape[-2]   # window_size_K (Q)

        if not k.shape[-1] == embed_dim or not v.shape[-1] == embed_dim:
            raise ValueError("Assume identical embedding dim for q, k, v")

        q = q.reshape(num_queries, bsz, embed_dim) # q: (window_size_K, batch*stock_num, hidden_size)
        # v = v.unsqueeze(dim=1)  # v: (batch*stock_num, 1, window_size_K, hidden_size)

        # q = q.unsqueeze(dim=-2)

        # convert to n_dim == 4
        # if num_queries == 1:
        #     k = k.unsqueeze(dim=1)
        #     v = v.unsqueeze(dim=1)

        # [Q * (B*I) * E_lstm] * [E_lstm * att_dim] -> [Q * (B*I) * att_dim]
        trans_q = self.query_transform(q)
        # [(B*I) * E_lstm] * [E_lstm * att_dim] -> [(B*I) * att_dim]
        trans_k = self.key_transform(k)

        # broadcast to (Q, B*I, att_dim)
        attn_input = trans_k + trans_q

        # transfer to (Q * (B*I), att_dim)
        attn_input = attn_input.view(-1, attn_input.shape[-1])

        # softmaxed ((Q * (B*I)), 1)
        q_score = self.attn_score(attn_input)
        # transfer to (B*I, Q, 1)
        q_score = q_score.view(bsz, num_queries, -1)

        # broadcasting [(B*I)*Q*1] * [(B*I)*1*E_lstm] -> [(B*I)*Q*E_lstm]

        #  [(B*I)*Q*1] * [(B*I)*Q*E_lstm] -> [(B*I)*Q*E_lstm]
        self_attn_rep = torch.multiply(q_score, v)

        # [(B*I) * Q * E_lstm] -> [(B*I) * 1 * E_lstm]
        summed_ha_rep = torch.sum(self_attn_rep, dim=-2)

        return summed_ha_rep

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
        self.ha = StdAttn(input_size=lstm_h_dim,
                          att_dim=attn_w_dim, dropout=dropout)

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
        # 这里outputs�?batch*stock_num, window_size_K, hidden_size)

        outputs, _ = self.lstm(x)
        out_rep = self.ha(outputs, outputs[:,-1,:], outputs)

        out_rep = out_rep.view(batch_size, num_stock, -1)
        # out_rep的形状：(batch_size, stock_num, hidden_size)

        return out_rep

class BasicCANN(nn.Module):
    def __init__(self, num_heads=CANN_HEADS, embed_dim=LSTM_H_DIM):
        super().__init__()
        self.attn = nn.MultiheadAttention(num_heads=num_heads,
                                          embed_dim=embed_dim,)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid())

    def forward(self, x):
        """
        :param x: B*I*E_c
        :return: B*I*1
        """
        # B * I * E -> B * I * E
        cross_attn_rep, attn_weights = self.attn(x, x, x)
        # B * I * E -> B * I * 1
        scores = self.score(cross_attn_rep)
        return scores, attn_weights

class PortfolioGenerator(nn.Module):
    # 固定规则，无需学习
    def __init__(self, G=STOCK_POOL):
        super().__init__()
        self.G = G

    def forward(self, x):
        """
        :param x: B*I
        :return:portfolio: B*I
        """
        # remove redundant last dim of size 1
        x = x.squeeze(-1)
        if x.shape[-1] <= 2 * self.G:
            raise ValueError(f"len of configurable stocks:{2 * self.G} "
                             f"> available stocks:{x.shape[-1]}")

        # bsz * num_asset
        # b_c = torch.zeros_like(x, requires_grad=True)

        # batch-wise ranking & binarize
        sorted_x, sorted_indices = torch.sort(x, dim=-1, descending=True)

        winner_assets_x = sorted_x[:, :self.G]
        # winner_assets_indices = sorted_indices[:, :self.G]
        loser_assets_x = sorted_x[:, -self.G:]
        # loser_assets_indices = sorted_indices[:, :self.G]

        # --- Todo: temp: the following codes suit only bsz = 1 -----#
        winner_proportion = F.softmax(winner_assets_x, dim=-1)
        loser_proportion = -F.softmax(1 - loser_assets_x, dim=-1)

        zeros_assets = torch.zeros_like(sorted_x[:, self.G:-self.G],
                                        requires_grad=True).type(x.dtype)

        b_c = torch.cat([winner_proportion, zeros_assets,
                         loser_proportion], dim=1)

        return b_c, sorted_indices

class PGNetwork(nn.Module):
    def __init__(self):
        super(PGNetwork, self).__init__()
        self.lstm_ha = LSTMHA()
        self.cann = BasicCANN()
        self.portfolio_gen = PortfolioGenerator()

    def forward(self, x):
        if x.dim() == 3:
            x_b = x.unsqueeze(0)  # TODO: 第一个维度给batch
        else:
            x_b = x

        # x_b: (batch_size, num_stock, window_size_K, feature_dim)
        ha_rep = self.lstm_ha(x_b)
        # ha_rep: (batch_size, num_stock, E_c)
        cann_score, attn_w = self.cann(ha_rep)
        # cann_score: (batch_size, num_stock, 1), attn_w: (batch_size, num_stock, 1)
        portfolios, sorted_indices = self.portfolio_gen(cann_score)
        return portfolios, sorted_indices

class PG(object):
    # dqn Agent
    def __init__(self, env):
        print("initialize")
        # init network parameters
        # self.network = PGNetwork().to(device)
        self.network = torch.load('modelsave/all_net_3000.pkl')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        # TODO: 解决输入state维度问题
        observation = torch.FloatTensor(observation).to(device)
        network_output, sorted_indices = self.network.forward(observation)

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

        return action

def main():
    env = ALPHA_ENV(param)
    agent = PG(env)
    for test_epi in range(100):
        # Test every TEST_EPI episodes
        #if episode % TEST_EPI == 0:
        total_reward = 0
        for i in range(10):
            state = env.reset()
            for j in range(MAX_STEP):
                action = agent.choose_action(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('Evaluation | episode: ', test_epi, ' | Evaluation Average Reward:', ave_reward)



if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
