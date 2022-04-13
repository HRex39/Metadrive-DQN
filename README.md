# Metadrive-DQN
Metadrive-simulator Agent using DQN  

## PreRequisite
1. [MetaDrive](https://metadrive-simulator.readthedocs.io/en/latest/install.html)
2. [Pytorch](https://pytorch.org/)
3. Tensorboard(pip install tensorboard)

## Usage
1. train
```
python3 train.py
tensorboard --logdir=./log --port <your port>
```

2. play
```
<change loading path in play.py>
python3 play.py
```
