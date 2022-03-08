from env_maze import Maze
from q_learning import QLearningTable

env = Maze()
RL = QLearningTable(actions=list(range(env.n_actions)), learning_rate=0.1)

for episode in range(10):
    state = env.reset()
    while True:
        env.render()  # update env
        action = RL.choose_action(str(state))  # choose action
        state_, reward, done = env.step(action)  # take action and get next state and reward
        RL.learn(str(state), action, reward, str(state_))  # learn
        state = state_
        if done:
            break

print(RL.q_table)



