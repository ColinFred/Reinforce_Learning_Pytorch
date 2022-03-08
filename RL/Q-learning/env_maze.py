import time
from random import choice


class Maze:
    def __init__(self):
        self.action_space = ['l', 'r']
        self.n_actions = len(self.action_space)
        self.states = range(6)
        self.rewards = [0, 0, 0, 0, 0, 1]
        self.current_state = 0

    def render(self):
        env = list('-----T')
        if self.current_state != self.states[-1]:
            env[self.current_state] = 'o'
        print('\r{}'.format(''.join(env)), end='')
        time.sleep(0.5)

    def step(self, action):
        if self.current_state == self.states[0] and action == 0:  # can't turn left
            pass
        elif action == 0:  # turn left
            self.current_state -= 1
        elif action == 1:  # turn right
            self.current_state += 1

        reward = self.rewards[self.current_state]
        done = False
        if self.current_state == self.states[-1]:  # done
            s_ = "terminal"
            done = True
        else:
            s_ = self.current_state

        return s_, reward, done

    def reset(self):
        self.current_state = 0
        self.render()
        return self.current_state


if __name__ == "__main__":
    maze = Maze()
    s = maze.reset()
    for i in range(10):
        action = choice([0, 1])
        s_, reward, done = maze.step(action)
        maze.render()
        s = s_
