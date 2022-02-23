import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

param = {
    # training parameters
    'trainDt_path': 'zztestn.csv',
    'testDt_path': 'zztestn.csv',
    'TRAIN': False,
    'TEST': True,
    'train_episode': 15000,
    'lr': 0.01,
    'save_model_name': 'TSA_modelsave/18train/ts30_lstmHA',
    'test_model_name': 'TSA_modelsave/18train/ts30_lstmHA_final',
    'result_path': 'TSA_result/18train19test/res30_lstmHA_final',
    'img_path': 'TSA_img/18train19test_ts30_lstmHA_stock0',
    'save_freq': 1000,

    # adjusting parameters
    'K_lookback': 12,
    'batch_size': 48,
    'future_ts': 3,
    'TEST_NUM_STOCK': 25,

    # network parameters
    'feature_dim': 31,
    'lstm_h_dim': 128,
    'attn_w_dim': 64,
    'lstm_dropout': 0.2,
    'lstm_layer': 1,

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader():
    def __init__(self, param, isTrain):
        self.isTrain = isTrain
        if self.isTrain == True:
            df = pd.read_csv(param['trainDt_path'])
        else:
            df = pd.read_csv(param['testDt_path'])

        # self.data_train = df.drop(['CLOSE_AFTER'], axis=1)    # 这里把history price也一起送到feature里
        self.data_train = df
        self.close_train = df['CLOSE_AFTER']
        self.K = param['K_lookback']
        self.batch_size = param['batch_size']   # in trading this means 'seq_time'
        self.future_ts = param['future_ts']
        self.time_stump = df['time']
        self.stock_all = df['thscode'].unique()  # len=454
        self.stock_list = df['thscode']

        self.test_stock_num = param['TEST_NUM_STOCK']
        self.all_stock_close = []   # (stock_num, time_step)
        self.all_stock_feature = [] # (stock_num, time_step, feature_dim)

        for i in range(self.test_stock_num):
            thscode = self.stock_all[i]
            dt = self.data_train[self.stock_list == thscode]
            stock_i_feature = np.array(dt.iloc[:, 2:])      # TODO: pay attention to the feature dimension
            self.all_stock_feature.append(stock_i_feature)
            self.all_stock_close.append(self.close_train[self.stock_list == thscode])

        self.all_stock_close = np.array(self.all_stock_close).transpose((1,0)) # (time_step, stock_num)
        self.all_stock_feature = np.array(self.all_stock_feature).transpose((1,0,2)) # (time_step, stock_num, feature_dim)

    def create_train_batch(self):
        batch_X = []
        batch_y = []
        self.time_stump1 = self.time_stump[self.stock_list == self.stock_all[0]]
        # TODO: the length of the trained time series should be at least K+batchsize+futurets
        if len(self.time_stump1) < self.K + self.batch_size + self.future_ts:
            raise ValueError("Time series too short!")

        self.trade_date = np.random.randint(self.K - 1, len(self.time_stump1) - self.batch_size - self.future_ts)  # randomly pick a start point for trading
        for bs in range(self.batch_size):
            # 每次循环中，在history里面随机选一个起始点并向前推移seq=batch_size步构造一个训练样本
            # feature for all stock at trade_date+t and previous K step
            # all_stock_feature_in_K: (window_size_K, stock_num, feature_dim)
            all_stock_feature_in_K = self.all_stock_feature[self.trade_date + bs - self.K + 1: self.trade_date + bs + 1]
            # all_stock_his_state: (num_stock, window_size_k, feature_dim)
            all_stock_his_state = all_stock_feature_in_K.transpose((1, 0, 2))
            stock_price = self.all_stock_close[self.trade_date + bs + self.future_ts]  # price vector (all stock) at self.trade_date+t

            batch_X.append(all_stock_his_state)
            batch_y.append(stock_price)

        return batch_X, batch_y

    def create_test(self):
        test_X = []
        true_y = []
        self.time_stump1 = self.time_stump[self.stock_list == self.stock_all[0]]
        if len(self.time_stump1) < self.K + self.future_ts:
            raise ValueError("Time series too short!")

        self.trade_date = self.K - 1
        for s in range(len(self.time_stump1) - self.future_ts - self.K):
            all_stock_feature_in_K = self.all_stock_feature[self.trade_date + s - self.K + 1: self.trade_date + s + 1]
            # all_stock_his_state: (num_stock, window_size_k, feature_dim)
            all_stock_his_state = all_stock_feature_in_K.transpose((1, 0, 2))
            stock_price = self.all_stock_close[self.trade_date + s + self.future_ts]  # price vector (all stock) at self.trade_date+t

            test_X.append(all_stock_his_state)
            true_y.append(stock_price)

        return test_X, true_y

class LSTMHA(nn.Module):
    def __init__(self, lstm_input_size=param['feature_dim'], lstm_h_dim=param['lstm_h_dim'],
                 lstm_num_layers=param['lstm_layer'],
                 attn_w_dim=param['attn_w_dim'], dropout=param['lstm_dropout']):
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
        self.ll = nn.Linear(param['K_lookback'], 1)

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

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.lstm_ha = LSTMHA()
        self.ll = nn.Linear(param['lstm_h_dim'], 1)

    def forward(self, x_b):
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
        return stock_score

class Optimizer(object):
    def __init__(self):
        self.network = Network().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=param['lr'])

    def update_parameters(self, b_X, b_y):
        network_output = self.network.forward(torch.FloatTensor(b_X).to(device))
        # network_output: (batch_size, num_stock)
        target = torch.LongTensor(b_y).float()
        # target: (num_stock)

        loss = F.mse_loss(network_output, target.to(device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

def metrics(true_vec, pred_vec):
    num_series = len(true_vec.columns)
    mse = [(np.square(true_vec[[i]] - pred_vec[[i]]).sum()) / len(true_vec[[i]]) for i in range(num_series)]
    mae = [np.abs(true_vec[[i]] - pred_vec[[i]]).sum() / len(true_vec[[i]]) for i in range(num_series)]
    print("Average mse:", np.mean(mse))
    print("Average mae:", np.mean(mae))

def tsplot(true_vec, pred_vec):
    xticks = np.arange(len(true_vec))
    plt.figure(figsize=(16,5))
    plt.plot(xticks, true_vec[[0]], 'black', lw=2, label='actual price')
    plt.plot(xticks, pred_vec[[0]], 'blue', lw=2, label='predicted price')
    plt.grid()
    plt.legend()
    plt.title('Stock prediction')
    plt.xlabel('Time')
    plt.savefig(param['img_path']+'.jpg',dpi=600)
    #plt.show()

def train():
    dataLoader = DataLoader(param, isTrain=True)
    optim = Optimizer()
    aveL, minL = 0, np.inf
    for episode in range(param['train_episode']):
        batch_X, batch_y = dataLoader.create_train_batch()
        aveL += optim.update_parameters(batch_X, batch_y)
        if episode % 100 == 0:
            print("Train epi:", episode, " | Ave loss:", aveL/100)
            if aveL <= minL:
                torch.save(optim.network, param['save_model_name']+'_best.pkl')
                print("Saving best reward at episode ", episode)
                minL = aveL
        aveL = 0
        if episode % param['save_freq'] == 0:
            torch.save(optim.network, param['save_model_name']+'_'+str(episode)+'.pkl')
    torch.save(optim.network, param['save_model_name']+'_final.pkl')

def test():
    dataLoader = DataLoader(param, isTrain=False)
    test_X, true_y = dataLoader.create_test()
    true_y = np.array(true_y)
    network = torch.load(param['test_model_name']+'.pkl')
    pred_y = network.forward(torch.FloatTensor(test_X).to(device)).detach().cpu().numpy()
    # dim of true_y/pred_y: (test_time, num_stock)
    # 考虑把真实值和预测值都存储到dataFrame中
    true_price = pd.DataFrame(true_y)
    pred_price = pd.DataFrame(pred_y)
    metrics(true_price, pred_price)
    tsplot(true_price, pred_price)

    true_price.to_csv(param['result_path'] + '_true.csv',index=False)
    pred_price.to_csv(param['result_path'] + '_pred.csv', index=False)

if __name__ == '__main__':
    time_start = time.time()
    if param['TRAIN']:
        train()
    if param['TEST']:
        test()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
