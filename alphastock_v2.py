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
MODEL_PATH = 'modelsave/all_net_best.pkl'
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
        # q, v: (batch*stock_num, window_size_K, hidden_size)
        # k: (batch*stock_num, hidden_size)
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
        Receive the fixed-length (window size K) sequence as input
        """
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
        out_rep = self.ha(outputs, outputs[:,-1,:], outputs)

        out_rep = out_rep.view(batch_size, num_stock, -1)
        # out_rep: (batch_size, stock_num, hidden_size)

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
    # Fixed rule, without learnable parameters
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
        # sorted_winner_assets_x = winner_assets_x.sort_values(by=sorted_indices)
        # sorted_loser_assets_x = loser_assets_x.sort_values(by=sorted_indices)

        b_c = torch.cat([winner_proportion, zeros_assets,
                         loser_proportion], dim=1)

        # convert back to orig order
        # Todo: to fix : inplace variables -> prevent differentiating
        # b_c[:, winner_assets_indices] = winner_proportion
        # b_c[:, loser_assets_indices] = loser_proportion

        # zeros_like: 0 tensor with same dim in '()'
        # reordered_b_c = torch.zeros_like(b_c, requires_grad=True).type(x.dtype)

        # out-of-place copy,  but only support one dim index
        # reordered_b_c = reordered_b_c.scatter(
        #               index=sorted_indices,
        #               src=b_c,
        #               dim=1)

        return b_c, sorted_indices

class PGNetwork(nn.Module):
    def __init__(self):
        super(PGNetwork, self).__init__()
        self.lstm_ha = LSTMHA()
        self.cann = BasicCANN()
        self.portfolio_gen = PortfolioGenerator()

    def forward(self, x):
        if x.dim() == 3:
            x_b = x.unsqueeze(0)  # TODO: The first dimension is given to 'batch'
        else:
            x_b = x

        # x_b: (batch_size, num_stock, window_size_K, feature_dim)
        ha_rep = self.lstm_ha(x_b)
        # ha_rep: (batch_size, num_stock, E_c)
        cann_score, attn_w = self.cann(ha_rep) # TODO: collapse to 0 or 1 after attention??

        # cann_score: (batch_size, num_stock, 1), attn_w: (batch_size, num_stock, 1)
        portfolios, sorted_indices = self.portfolio_gen(cann_score)
        return portfolios, sorted_indices

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
        network_output, sorted_indices = self.network.forward(observation)

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
        softmaxInput, _ = self.network.forward(torch.FloatTensor(self.ep_obs).to(device))

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
        network_output, sorted_indices = self.network.forward(observation)

        action_prob = network_output.detach().cpu().numpy()[0]
        action_shuffle = [np.random.binomial(1,i)
                          if i>=0 else -np.random.binomial(1,-i)
                          for i in action_prob
                          ]
        action = [0]*len(action_shuffle)
        sorted_indices = sorted_indices.detach().cpu().numpy()[0]
        for i in range(len(action_shuffle)):
            action[sorted_indices[i]] = action_shuffle[i]
        return action

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
                torch.save(agent.network, 'modelsave/all_net_best.pkl')
                print("Saving best reward at episode ", episode)
                max_train_reward = ave_reward
            statistic.append(ave_reward)

        if episode % SAVE_FREQ == 0:
            torch.save(agent.network, 'modelsave/all_net_'+str(episode)+'.pkl')

    result = pd.DataFrame(statistic)
    #result.to_csv('aveReward_full.csv')

def test():
    env = ALPHA_ENV(param, isTrain=False)
    agent = PGtest(MODEL_PATH)
    total_reward = 0
    state = env.reset()
    for j in range(MAX_STEP):
        action= agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            result = pd.DataFrame({
                'portfolio': env.portfolio_list,
                'buyhold': env.buyhold_list,
            })
            result.to_csv('result_all.csv',index=False)
            break

if __name__ == '__main__':
    time_start = time.time()
    if param['TRAIN']:
        train()
    if param['TEST']:
        test()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
