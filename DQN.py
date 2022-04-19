'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>
'''

# Net
from DQN_Net import *

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tensorboard
from torch.utils.tensorboard import SummaryWriter   

# Numpy
import numpy as np
from collections import deque

# Hyper Parameters
BATCH_SIZE = 32
LR = 2e-4                   # learning rate
EPSILON = 0.1               # greedy policy
SETTING_TIMES = 500         # greedy setting times 
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 20000

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class DQN(object):
    def __init__(self, is_train=True):
        self.IS_TRAIN = is_train
        self.eval_net, self.target_net = DQN_Net().to(device), DQN_Net().to(device)
        self.learn_step_counter = 0     # for target updating
        self.memory_counter = 0         # for storing memory
        # (s,a,r,s_)一组4个数据
        self.memory = deque(range(MEMORY_CAPACITY*4),maxlen=MEMORY_CAPACITY*4)     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./log') if self.IS_TRAIN else None

        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPSILON = EPSILON if self.IS_TRAIN else 1.0
        self.SETTING_TIMES = SETTING_TIMES
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.N_STATES = self.eval_net.N_STATES

    # 此时只选择action的序号，具体的action放在主函数中确定
    def choose_action(self, x):

        stateinput = x[None,:] # add 1 dimension to input state x
        x = torch.from_numpy(stateinput).to(device) 

        if np.random.uniform() < self.EPSILON: # greedy
            action_value = self.eval_net.forward(x).cpu()
            # torch.max() 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
            action_index = torch.max(action_value, 1)[1].data.numpy()[0] # 此时已经转变为index的形式
            action_max_value = torch.max(action_value, 1)[0].data.numpy()[0]
        else:
            action_index = np.random.randint(0, self.eval_net.N_ACTIONS)
            action_max_value = 0
        return action_index, action_max_value

    # store memory
    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index*4] = s
        self.memory[index*4+1] = a
        self.memory[index*4+2] = r
        self.memory[index*4+3] = s_
        self.memory_counter += 1
    
    def learn(self):
        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        # TODO: Concatenate
        b_s = np.empty((1,4,80,80), dtype=np.float32).fill(np.nan)
        b_a = np.empty(0, dtype=np.int32).fill(np.nan)
        b_r = np.empty(0, dtype=np.float32).fill(np.nan)
        b_s_ = np.empty((1,4,80,80), dtype=np.float32).fill(np.nan)
        b_empty = False

        for i in sample_index:
            b_s = np.concatenate((b_s, np.expand_dims(self.memory[i*4], axis=0)), axis=0) if b_empty else np.expand_dims(self.memory[i*4], axis=0)
            b_a = np.concatenate((b_a, np.expand_dims(self.memory[i*4+1], axis=0)), axis=0) if b_empty else np.expand_dims(self.memory[i*4+1], axis=0)
            b_r = np.concatenate((b_r, np.expand_dims(self.memory[i*4+2], axis=0)), axis=0) if b_empty else np.expand_dims(self.memory[i*4+2], axis=0)
            b_s_ = np.concatenate((b_s_, np.expand_dims(self.memory[i*4+3], axis=0)), axis=0) if b_empty else np.expand_dims(self.memory[i*4+3], axis=0)
            b_empty = True

        b_a = np.expand_dims(b_a, axis=0)
        b_r = np.expand_dims(b_r, axis=0)

        b_s = torch.from_numpy(b_s).to(device)
        b_a = torch.from_numpy(b_a).to(device)
        b_r = torch.from_numpy(b_r).to(device).to(torch.float32)
        b_s_ = torch.from_numpy(b_s_).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a.t())  # dim=1是横向的意思 shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r.view(BATCH_SIZE, 1) + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        
        loss.backward()
        self.optimizer.step()
        if (self.IS_TRAIN):
            if (self.learn_step_counter % 100000 == 0):
                self.writer.add_scalar('Loss', loss.cpu(), self.learn_step_counter)

    def save(self,path):
        torch.save(self.eval_net.state_dict(), path)
    def load(self,path):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


