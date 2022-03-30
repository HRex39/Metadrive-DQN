'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>
'''

# Metadrive
from metadrive import MetaDriveEnv

# Other Lib
import numpy as np
import random
from DQN import *
from math import floor

# Init Metadrive Env
config = dict(
        use_render=True,
        traffic_density=0.0,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        use_lateral=True,
        # map="SCrROXT",
        map="OCXT",
        # start_seed=random.randint(0, 1000),
        # map=4,  # seven block
        # environment_num=100,
    )

def choose_steering(action_index):
    steering_index = floor(action_index / 3)
    switch = {  0:-0.5,
                1:0.0,
                2:0.5,}
    steering = switch.get(steering_index)
    return steering

def choose_acceleration(action_index):
    acceleration_index = floor(action_index % 3)
    switch = {  0:-0.5,
                1:0.0,
                2:0.5,}
    acceleration = switch.get(acceleration_index)
    return acceleration

if __name__ == '__main__':
    env = MetaDriveEnv(config)
    dqn = DQN(is_train=False)
    dqn.load("./check_points_swing.tar")
    print('--------------\nLoading experience...\n--------------')

    for i_episode in range(100000):
        s = env.reset()
        s = s[: dqn.N_STATES] 
        env.vehicle.expert_takeover = True
        total_reward = 0

        while True:
            env.render(
                text={
                    "score": total_reward,
                }
            )
            # take action based on the current state
            action_index, action_value = dqn.choose_action(s)
            # choose action
            steering = choose_steering(action_index)
            acceleration = choose_acceleration(action_index)
            action = np.array([steering, acceleration])

            print('\r' + "steering: "+str(steering)+"acce: "+str(acceleration) , end='', flush=True)
            # obtain the reward and next state and some other information
            s_, reward, done, info = env.step(action)
            # slice s and s_
            s = s[: dqn.N_STATES] 
            s_ = s_[: dqn.N_STATES]
            total_reward += reward
            s = s_  
            if done and info["arrive_dest"]:
                print('\nEp: ', i_episode, ' |', 'Ep_r: ', round(total_reward, 2))
                break

