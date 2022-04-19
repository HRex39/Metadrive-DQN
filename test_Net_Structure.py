'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>
'''

from torch.utils.tensorboard import SummaryWriter
import numpy as np

from DQN_Net import *

writer = SummaryWriter()
model = DQN_Net()
input = np.random.rand(80,80,4)
input = input.transpose(2,1,0)
stateinput = input[None,:]
x = torch.from_numpy(stateinput).to(torch.float32)
writer.add_graph(model, x)
writer.close()