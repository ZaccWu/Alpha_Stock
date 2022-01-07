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

# Hyper Parameters for PG Network
GAMMA = 0.95  # discount factor
LR = 0.01  # learning rate

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
        # k是(batch*stock_num, hidden_size)
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
    def __init__(self, lstm_input_size=29, lstm_h_dim=32,
                 lstm_num_layers=1,
                 attn_w_dim=64, dropout=0.2):
        """
        接收窗口大小为K的固定长度序列作为输入
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
        x的形状：batch_size * stock_num * window_size_K * feature_dim (ndim=4)
        """
        # if is aligned batch inputs (B * I * T * E_feat), ndim=4
        batch_size = x.shape[0]
        num_stock = x.shape[1]

        # lstm设置为batch_first则需要输入(batch, seq_len, input_size)的数据
        # 在这里让x变为(batch_size * stock_num, window_size_K, feature_dim)的张量
        x = x.view(batch_size * num_stock, x.shape[-2], x.shape[-1])

        # 设置了batch_fisrt=True后，单向lstm输出outputs(batch, seq_len, hidden_size), (hn, cn)
        # 这里outputs是(batch*stock_num, window_size_K, hidden_size)

        outputs, _ = self.lstm(x)
        out_rep = self.ha(outputs, outputs[:,-1,:], outputs)

        out_rep = out_rep.view(batch_size, num_stock, -1)

        return out_rep

class BasicCANN(nn.Module):
    def __init__(self, num_heads=4, embed_dim=32):
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
    def __init__(self, G=7):
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

        # zeros_like: 生成和括号内变量维度一致的全是0的内容
        reordered_b_c = torch.zeros_like(b_c, requires_grad=True).type(x.dtype)


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
            x_b = x.unsqueeze(0)  # TODO: 第一个维度给batch
        else:
            x_b = x

        # x_b: (batch_size, num_stock, window_size_K, feature_dim)
        ha_rep = self.lstm_ha(x_b)
        # ha_rep: (batch_size, num_stock, E_c)
        cann_score, attn_w = self.cann(ha_rep) # TODO: attention之后塌缩成0和1

        # cann_score: (batch_size, num_stock, 1), attn_w: (batch_size, num_stock, 1)
        portfolios, sorted_indices = self.portfolio_gen(cann_score)
        return portfolios, sorted_indices

    # def initialize_weights(self):
    #     for m in self.modules():
    #         nn.init.normal_(m.weight.data, 0, 0.1)
    #         nn.init.constant_(m.bias.data, 0.01)
    #         # m.bias.data.zero_()

class PG(object):
    # dqn Agent
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        # self.state_dim = env.observation_space.shape[0]
        # self.action_dim = env.action_space.n

        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        self.network = PGNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        # TODO: 解决输入state维度问题
        observation = torch.FloatTensor(observation).to(device)
        network_output, sorted_indices = self.network.forward(observation)

        action_prob = network_output.detach().cpu().numpy()[0]
        # action_prob的形式: [0.4, 0.1, 0,..., 0.2, 0.3]
        # action_prob=np.array([0.4,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.3])

        action_shuffle = [np.random.binomial(1,i)
                          if i>=0 else -np.random.binomial(1,-i)
                          for i in action_prob
                          ] # 按概率决定是否交易
        # action_shuffle的形式: [1, 0, 0,..., -1, 0]（随机向量）

        action = [0]*len(action_shuffle)
        sorted_indices = sorted_indices.detach().cpu().numpy()[0]

        for i in range(len(action_shuffle)):
            action[sorted_indices[i]] = action_shuffle[i]

        # 返回一个0-1action list，代表这支股票是否需要交易
        # TODO: 假设前面的（多头）为正，后面的（空头）为负，但向量绝对值之和归一
        # TODO: 在RL优化当中，可能只能让action按概率取某一个值，而不能是一个向量

        return action

    # 将状态，动作，奖励这一个transition保存到三个列表中
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def update_parameters(self):
        self.time_step += 1

        # Step 1: 计算每一步的状态价值
        epiRewardDiscounted = np.zeros_like(self.ep_rs)
        running_add = 0
        # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            epiRewardDiscounted[t] = running_add

        epiRewardDiscounted = np.array(epiRewardDiscounted,dtype=np.float64)
        epiRewardDiscounted -= np.mean(epiRewardDiscounted)  # 减均值
        epiRewardDiscounted /= np.std(epiRewardDiscounted)  # 除以标准差
        epiRewardDiscounted = torch.FloatTensor(epiRewardDiscounted).to(device)

        # Step 2: 前向传播
        softmaxInput, _ = self.network.forward(torch.FloatTensor(self.ep_obs).to(device))

        tar = torch.LongTensor(self.ep_as).float()
        # TODO: 输入的target，在本机上提示需要float的类型，但是服务器上却要long

        negLogProb = F.cross_entropy(input=softmaxInput, target=tar.to(device),
                                       reduction='none')
        # TODO: cross_entropy的问题，注意这里需要torch版本>=1.10.0才有概率式target的支持

        # Step 3: 反向传播
        loss = torch.mean(negLogProb * epiRewardDiscounted)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次学习完后清空数组
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

# ---------------------------------------------------------
# Hyper Parameters
MAX_EPISODE = 15000  # Episode limitation
MAX_STEP = 1000  # Step limitation in an episode
TEST = 5  # The number of experiment test every 100 episode


def main():
    env = ALPHA_ENV()
    agent = PG(env)
    statistic = []
    for episode in range(1, MAX_EPISODE):
        state = env.reset() # 返回的state维数: (num_stock, window_size_k, feature_dim)

        # TODO: 解决batch问题，当前只适用于batch_size=1
        for step in range(MAX_STEP):
            action = agent.choose_action(state)  # softmax概率选择action
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)  # 新函数 存取这个transition
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                # print("episode: ", episode, "episode reward: {:.2f}".format(np.mean(agent.ep_rs)))
                agent.update_parameters()  # 更新策略网络
                break

        # Test every 100 episodes
        if episode % 100 == 0:
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
            statistic.append(ave_reward)

    result = pd.DataFrame(statistic)
    result.to_csv('aveReward.csv')



if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
