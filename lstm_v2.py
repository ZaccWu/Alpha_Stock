import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
from collections import deque
from env.alpha_env_v2 import ALPHA_ENV

# Hyper Parameters
MAX_EPISODE = 2000  # Episode limitation
MAX_STEP = 1000  # Step limitation in an episode
TEST = 5  # The number of experiment test every 100 episode
TEST_EPI = 100 # How often we test
SAVE_FREQ = 1000    # How often we save model

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
    'TEST_NUM_STOCK': 25,       # how many stock we choose
    # file path:
    # 'DT_PATH': 'combine/2018_zz500.csv',
    'DTRAIN_PATH': 'zztestn.csv',
    'DTEST_PATH': 'zztestn.csv',
    'TRAIN': True,  # is train
    'TEST': True,   # is test
}

# Parameters depend on your data
FEATURE_DIM = 29    # the feature of a stock
MODEL_PATH = 'modelsave/lstm_net_best.pkl'
# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMHA(nn.Module):
    def __init__(self, lstm_input_size=FEATURE_DIM, lstm_h_dim=LSTM_H_DIM,
                 lstm_num_layers=LSTM_LAYER,
                 attn_w_dim=ATTN_W_DIM, dropout=LSTM_DROPOUT):
        """
        Receive the fixed-length (window size K) sequence as input
        """
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
        x：batch_size * stock_num * window_size_K * feature_dim (ndim=4)
        """
        # if is aligned batch inputs (B * I * T * E_feat), ndim=4
        batch_size = x.shape[0]
        num_stock = x.shape[1]

        # if we set lstm 'batch_first=True', we need input: (batch, seq_len, input_size)
        # here, x is converted to tensor (batch_size * stock_num, window_size_K, feature_dim)
        x = x.view(batch_size * num_stock, x.shape[-2], x.shape[-1])

        # if we set lstm 'batch_fisrt=True', it outputs: (batch, seq_len, hidden_size), (hn, cn)

        outputs, _ = self.lstm(x)   # outputs: (batch*stock_num, window_size_K, hidden_size)
        outputs = outputs.transpose(1,2)  # (batch*stock_num, hidden_size, window_size_K)

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
        self.ll = nn.Linear(LSTM_H_DIM, 1)

    def forward(self, x):
        if x.dim() == 3:
            x_b = x.unsqueeze(0)  # TODO: The first dimension is given to 'batch'
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
    def __init__(self, env):
        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        self.network = PGNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        # TODO: fix the input state dim problem
        observation = torch.FloatTensor(observation).to(device)
        network_output, sorted_indices, _ = self.network.forward(observation)

        action_prob = network_output.detach().cpu().numpy()[0]
        # action_prob: shape like [0.4, 0.1, 0,..., 0.2, 0.3]
        # action_prob=np.array([0.4,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.3])

        action_shuffle = [np.random.binomial(1,i)
                          if i>=0 else -np.random.binomial(1,-i)
                          for i in action_prob
                          ] # whether to trade depends on the probability
        # action_shuffle: random vector such as [1, 0, 0,..., -1, 0]

        action = [0]*len(action_shuffle)
        sorted_indices = sorted_indices.detach().cpu().numpy()[0]

        for i in range(len(action_shuffle)):
            action[sorted_indices[i]] = action_shuffle[i]

        # return a 0-1 action list
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def update_parameters(self):
        self.time_step += 1

        # Step 1: calculate every state value in each step
        epiRewardDiscounted = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            epiRewardDiscounted[t] = running_add

        epiRewardDiscounted = np.array(epiRewardDiscounted,dtype=np.float64)
        epiRewardDiscounted -= np.mean(epiRewardDiscounted)
        epiRewardDiscounted /= np.std(epiRewardDiscounted)
        epiRewardDiscounted = torch.FloatTensor(epiRewardDiscounted).to(device)

        # Step 2: forward passing
        softmaxInput, _, _ = self.network.forward(torch.FloatTensor(self.ep_obs).to(device))

        tar = torch.LongTensor(self.ep_as).float()

        negLogProb = F.cross_entropy(input=softmaxInput, target=tar.to(device),
                                       reduction='none')
        # TODO: cross_entropy requires torch>=1.10.0 to support probabilistic target

        # Step 3: backward
        loss = torch.mean(negLogProb * epiRewardDiscounted)
        self.optimizer.zero_grad()
        #print('before backward ---------------------------------------')
        #print(self.network.lstm_ha.lstm.weight_hh_l0.grad)
        loss.backward()
        #print('after backward ---------------------------------------')
        #print(self.network.lstm_ha.lstm.weight_hh_l0.grad)
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

class PGtest(object):
    def __init__(self, MODEL_PATH):
        self.network = torch.load(MODEL_PATH)

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        network_output, sorted_indices, factor = self.network.forward(observation)

        action_prob = network_output.detach().cpu().numpy()[0]
        action_shuffle = [np.random.binomial(1,i)
                          if i>=0 else -np.random.binomial(1,-i)
                          for i in action_prob
                          ]
        action = [0]*len(action_shuffle)
        sorted_indices = sorted_indices.detach().cpu().numpy()[0]
        for i in range(len(action_shuffle)):
            action[sorted_indices[i]] = action_shuffle[i]
        return action, factor

def train():
    env = ALPHA_ENV(param, isTrain=True)
    agent = PG(env)
    statistic = []
    max_train_reward = -np.inf
    for episode in range(1, MAX_EPISODE):
        state = env.reset() # state: (num_stock, window_size_k, feature_dim)

        # TODO: only fits batch_size=1
        for step in range(MAX_STEP):
            action = agent.choose_action(state)  # softmax probability to choose action
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                # print("episode: ", episode, "episode reward: {:.2f}".format(np.mean(agent.ep_rs)))
                agent.update_parameters()
                break

        # Test every TEST_EPI episodes
        if episode % TEST_EPI == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(MAX_STEP):
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('Evaluation | episode: ', episode, ' | Evaluation Average Reward:', ave_reward)
            if ave_reward>max_train_reward:
                torch.save(agent.network, 'modelsave/lstm_net_best.pkl')
                print("Saving best reward at episode ", episode)
                max_train_reward = ave_reward
            statistic.append(ave_reward)

        if episode % SAVE_FREQ == 0:
            torch.save(agent.network, 'modelsave/lstm_net_'+str(episode)+'.pkl')

    result = pd.DataFrame(statistic)
    #result.to_csv('aveReward_full.csv')

def test():
    env = ALPHA_ENV(param, isTrain=False)
    agent = PGtest(MODEL_PATH)
    total_reward = 0
    factor_list = []
    state = env.reset()
    for j in range(MAX_STEP):
        action, factor = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        factor_list.append(factor.detach().cpu().numpy()[0])
        if done:
            result = pd.DataFrame({
                'portfolio': env.portfolio_list,
                'buyhold': env.buyhold_list,
            })
            result.to_csv('result_lstm.csv',index=False)
            break
    factor_list = pd.DataFrame(factor_list)
    factor_list.to_csv('factor_lstm.csv',index=False)

if __name__ == '__main__':
    time_start = time.time()
    if param['TRAIN']:
        train()
    if param['TEST']:
        test()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
