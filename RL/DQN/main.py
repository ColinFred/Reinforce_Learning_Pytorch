"""
from gym import envs
print(envs.registry.all())

ValuesView(├──CartPole: [ v0, v1 ]
├──MountainCar: [ v0 ]
├──MountainCarContinuous: [ v0 ]
├──Pendulum: [ v1 ]
├──Acrobot: [ v1 ]
├──LunarLander: [ v2 ]
├──LunarLanderContinuous: [ v2 ]
├──BipedalWalker: [ v3 ]
├──BipedalWalkerHardcore: [ v3 ]
├──CarRacing: [ v1 ]
├──Blackjack: [ v1 ]
├──FrozenLake: [ v1 ]
├──FrozenLake8x8: [ v1 ]
├──CliffWalking: [ v0 ]
├──Taxi: [ v3 ]
├──Reacher: [ v2 ]
├──Pusher: [ v2 ]
├──Thrower: [ v2 ]
├──Striker: [ v2 ]
├──InvertedPendulum: [ v2 ]
"""

from DQN import DQN
from doubleDQN import DoubleDQN
from duelingDQN import D3QN
from settings import *

import time
import gym

env = gym.make("CartPole-v1")
env = env.unwrapped

# print(env.action_space)  # number of action
# print(env.observation_space)  # number of state
# print(env.observation_space.high)
# print(env.observation_space.low)

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

RL = D3QN(n_action=NUM_ACTIONS, n_state=NUM_STATES, learning_rate=0.01)  # choose algorithm

total_steps = 0
for episode in range(1000):
    state, info = env.reset(return_info=True)
    ep_r = 0
    while True:
        env.render()  # update env
        action = RL.choose_action(state)  # choose action
        state_, reward, done, info = env.step(action)  # take action and get next state and reward
        x, x_dot, theta, theta_dot = state_  # change given reward
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2  # consider both locations and radians

        RL.store_transition(state, action, reward, state_)  # store transition
        RL.learn()  # learn

        ep_r += reward
        if total_steps % C == 0:  # every C steps update target_network
            RL.update_target_network()

        if done:
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2))
            break

        state = state_
        total_steps += 1
        time.sleep(0.05)
