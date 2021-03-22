import numpy as np        # useful for math operations and manipulating array

# possible ML frameworks
import tensorflow as tf
import pytorch
import jax



class DQNAgent:
    def __init__(self, q_network, optimizer):
        pass

    def get_actions(self, states, noisy=False):
        actions = self.q_network(states)
        if noisy:
            return some_noise(actions)
        else:
            return actions

    def train(self, states, q_targets):
        pass


class Buffer:
    def __init__(self, size):
        pass

    def add(self, data):
        pass

    def sample(self, batch_size):
        pass


class Procedure:
    def __init__(self, environment, agent, buffer):
        pass

    def gather_episode_data(self):
        done = False
        state = self.env.reset()
        data = []
        while not done:
            action = self.agent.get_actions(state)
            old_state = state
            new_state, reward, done, info = self.env.act(action)
            data.append((old_state, action, new_state, reward))
            state = new_state
        self.buffer.add(data)

    def evaluate(self, n_episodes):
        pass

    def train_from_buffer(self, batch_size):
        data = self.buffer.sample(batch_size)
        self.agent.train(???)

    def record_video(self, n_episodes):
        pass






q_network = ???
optimizer = ???
agent = DQNAgent(q_network, optimizer)

buffer_size = 10000
buffer = Buffer(buffer_size)

procedure = Procedure(environment, agent, buffer)


n_training_steps = 20000


while n_training_steps > 0:
    procedure.evaluate(n_episodes=10)
    for i in range(min(n_training_steps, 100)):
        procedure.gather_episode_data(???)
        procedure.train_from_buffer(batch_size=64)
        n_training_steps -= 1

print("Final performances:")
procedure.evaluate(n_episodes=100)
procedure.record_video(n_episodes=10)
