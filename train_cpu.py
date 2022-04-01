'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>

Reference: 
https://zhuanlan.zhihu.com/p/117287339
https://blog.csdn.net/qq_41871826/article/details/108263919
https://zhuanlan.zhihu.com/p/103630393
'''

# Metadrive
from metadrive import MetaDriveEnv

# Other Lib
import numpy as np
import random
from DoubleDQN import *
from math import floor

# Init Metadrive Env
config = dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.0,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        use_lateral=True,
        # map="SCrROXT",
        # map="OCXT",
        start_seed=random.randint(0, 1000),
        map=4,  # seven block
        environment_num=100,
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

'''
--------------Procedures of DQN Algorithm------------------
'''
if __name__ == '__main__':
    env = MetaDriveEnv(config)
    dqn = DoubleDQN(is_train=True)
    print('--------------\nCollecting experience...\n--------------')
    best_reward = 0

    for i_episode in range(100000):
        if i_episode <= dqn.SETTING_TIMES:
            dqn.EPSILON = 0.1 + i_episode / dqn.SETTING_TIMES * (0.9 - 0.1)
        s = env.reset()
        s = s[: dqn.N_STATES]
        env.vehicle.expert_takeover = True
        # indirect params
        total_reward = 0
        total_action_value = 0
        action_counter = 0 
        reward_counter = 0
        while True:
            # take action based on the current state
            action_index, action_value = dqn.choose_action(s)

            total_action_value += action_value
            if action_value != 0:
                action_counter += 1

            # choose action
            steering = choose_steering(action_index)
            acceleration = choose_acceleration(action_index)
            action = np.array([steering, acceleration])
            # step
            s_, reward, done, info = env.step(action)
            # slice s and s_
            s = s[: dqn.N_STATES] 
            s_ = s_[: dqn.N_STATES]
            # store the transitions of states
            dqn.store_transition(s, action_index, reward, s_)

            total_reward += reward
            reward_counter += 1
            
            # if the experience repaly buffer is filled, 
            # DQN begins to learn or update its parameters.       
            if dqn.memory_counter > dqn.MEMORY_CAPACITY:
                dqn.learn()
            if done:
                # if game is over, then skip the while loop.
                if best_reward <= total_reward:
                    best_reward = total_reward
                    dqn.save("./check_points.tar")
                if i_episode % 500 == 0:
                    dqn.save("./"+str(i_episode)+".tar")
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(total_reward, 2), ' |', 'Best_r: ', best_reward)
                break
            else:
                # use next state to update the current state. 
                s = s_
        dqn.writer.add_scalar('Ep_r', total_reward, i_episode)
        dqn.writer.add_scalar('Ave_r', total_reward/reward_counter, i_episode)
        dqn.writer.add_scalar('Ave_Q_value', total_action_value/action_counter, i_episode)


