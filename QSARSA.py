import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

class CliffWorld:

    def __init__(self, rows=4, columns=21, snakepit=False):
        self.rows = rows
        self.cols = columns
        self.pit = snakepit
        self.reset()

    def reset(self):
        self.state = (self.rows - 1, 0)

    def get_max_reward(self):
        return -1 * self.cols + 20

    def step(self, action):
        reward = -1

        if action == 0:  # Up
            next_state = (self.state[0] - 1, self.state[1])
        elif action == 1:  # Down
            next_state = (self.state[0] + 1, self.state[1])
        elif action == 2:  # Left
            next_state = (self.state[0], self.state[1] - 1)
        elif action == 3:  # Right
            next_state = (self.state[0], self.state[1] + 1)

        if (
                next_state[0] < 0 or next_state[0] >= self.rows
                or next_state[1] < 0 or next_state[1] >= self.cols
        ):
            next_state = self.state

        self.state = next_state
        terminated = False

        if self.pit and self.is_snakepit(self.state):
            reward = -100
            terminated = True

        if self.is_goal(self.state):
            reward = 20
            terminated = True

        if self.is_cliff(next_state):
            reward = -100
            terminated = True

        return self.state, reward, terminated

    def is_goal(self, state):
        # Goal State = (3, 20) == (self.rows-1,self.cols-1)
        return state == (self.rows - 1, self.cols - 1)

    def is_cliff(self, state):
        # Bottom row is cliff
        cliff = [(self.rows - 1, j) for j in range(1, self.cols - 1)]
        return state in cliff

    def is_snakepit(self, state):
        # Topmost = (0, self.cols - 1 / 2)
        return state == (0, 10)

    def show(self, history=None):
        actionMap = {-1: "S", 0: "^", 1: "v", 2: "<", 3: ">"}
        for i in range(0, self.rows):
            print('-------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, self.cols):
                # if self.state == (i,j):
                if (i, j) in history:
                    token = 'O'
                elif (i, j) == (self.rows - 1, 0):
                    token = 'S'
                elif self.is_goal((i, j)):
                    token = 'G'
                elif self.is_cliff((i, j)):
                    token = 'X'
                elif self.is_snakepit((i, j)) and self.pit:
                    token = 'P'
                else:
                    token = "."

                out += token + ' | '
            print(out)
        print('-------------------------------------------------------------------------------------')


# test with random policy
def test_env(max_steps=10, show=True):
    env = CliffWorld()
    env.reset()
    history = []
    history.append(env.state)
    r = 0
    for _ in range(max_steps):
        action = np.random.randint(0, 4)
        next_state, reward, terminated = env.step(action)
        history.append(next_state)
        r += reward
        if terminated:
            break

    print("total reward: ", r)
    if (show):
        env.show(history)


def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return np.random.randint(0, 4)
    else:
        return np.argmax(Q[state])

def test_policy(Q, max_steps=100, show=False, pit=False):
    env = CliffWorld(snakepit=pit)
    env.reset()
    history = []
    history.append(env.state)
    state = env.state[0] * env.cols + env.state[1]
    total_reward = 0
    for step in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, terminated = env.step(action)
        history.append(next_state)
        next_state = next_state[0] * env.cols + next_state[1]
        total_reward += reward
        state = next_state
        if terminated:
            break
    if show:
        env.show(history)
    return total_reward

def train_sarsa(max_episodes=1000, max_steps=100, alpha=0.1, gamma=0.9, epsilon=0.3, pit=False):
    env = CliffWorld(snakepit=pit)
    Q = np.zeros((env.rows * env.cols, 4))  # Q table initialized to all zeros for each state and action
    rewards_history = []
    converged = False

    for episode in range(max_episodes):
        env.reset()
        state = env.state[0] * env.cols + env.state[1]
        action = epsilon_greedy(Q, state, epsilon)
        reward_per_episode = 0
        for step in range(max_steps):
            next_state, reward, terminated = env.step(action)
            next_state = next_state[0] * env.cols + next_state[1]
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state = next_state
            action = next_action
            reward_per_episode += reward
            if terminated:
                break

        total_reward = test_policy(Q, max_steps)
        rewards_history.append(reward_per_episode)

        # break if the agent has solved the problem
        if total_reward == env.get_max_reward():
            if converged:
                break
            else:
                converged = True
        else:
            converged = False
    return Q, episode, rewards_history

def train_qlearning(max_episodes=1000, max_steps=100, alpha=0.1, gamma=0.9, epsilon=0.1, pit=False):
    env = CliffWorld(snakepit=pit)
    Q = np.zeros((env.rows * env.cols, 4))
    rewards_history = []
    converged = False

    for episode in range(max_episodes):
        env.reset()
        state = env.state[0] * env.cols + env.state[1]
        reward_per_episode = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated = env.step(action)
            next_state = next_state[0] * env.cols + next_state[1]
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state
            reward_per_episode += reward
            if terminated:
                break

        total_reward = test_policy(Q, max_steps)
        rewards_history.append(reward_per_episode)

        # break if the agent has solved the problem
        if total_reward == env.get_max_reward():
            if converged:
                break
            else:
                converged = True
        else:
            converged = False
    return Q, episode, rewards_history

def train_qlearning_replay(max_episodes=1000, buffer_size=100, max_steps=100, alpha=0.1, gamma=0.9, epsilon=0.1,
                           pit=False):
    env = CliffWorld(snakepit=pit)
    Q = np.zeros((env.rows * env.cols, 4))
    rewards_history = []
    replay_buffer = []
    converged = False

    for episode in range(max_episodes):
        env.reset()
        state = env.state[0] * env.cols + env.state[1]
        reward_per_episode = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated = env.step(action)
            next_state = next_state[0] * env.cols + next_state[1]
            replay_buffer.append((state, action, reward, next_state))
            state = next_state
            reward_per_episode += reward
            if terminated:
                break
        if len(replay_buffer) > buffer_size:
            replay_buffer = replay_buffer[-buffer_size:]
        for state, action, reward, next_state in replay_buffer:
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        total_reward = test_policy(Q, max_steps)
        rewards_history.append(reward_per_episode)
        # break if the agent has solved the problem
        if total_reward == env.get_max_reward():
            if converged:
                break
            else:
                converged = True
        else:
            converged = False
    return Q, episode, rewards_history

def Q1():
    # Q1. Policy of Sarsa, Q Learning and replay buffer
    Q_sarsa, _, _ = train_sarsa(max_episodes=2000, alpha=0.2, gamma=0.8, epsilon=0.2)
    Q_qlearning, _, _ = train_qlearning()
    Q_qlearning_replay, _, _ = train_qlearning_replay()

    print("Policy obtained under SARSA:")
    test_policy(Q_sarsa, show=True)
    print()
    print("Policy obtained under Q Learning:")
    test_policy(Q_qlearning, show=True)
    print()
    print("Policy obtained under Q Learning Replay:")
    test_policy(Q_qlearning_replay, show=True)
    print()

    mean_Q_sarsa = np.mean(Q_sarsa, axis=1)
    mean_Q_qlearning = np.mean(Q_qlearning, axis=1)
    mean_Q_qlearning_replay = np.mean(Q_qlearning_replay, axis=1)

    # Plot heatmap for each algorithm
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.title("SARSA")
    sns.heatmap(mean_Q_sarsa.reshape(4, 21))
    plt.subplot(3, 1, 2)
    plt.title("Q-learning")
    sns.heatmap(mean_Q_qlearning.reshape(4, 21))
    plt.subplot(3, 1, 3)
    plt.title("Q-learning with replay buffer")
    sns.heatmap(mean_Q_qlearning_replay.reshape(4, 21))
    # plt.savefig('policies_heatmap.png', bbox_inches='tight')
    plt.show()

def Q2():
    # Q2. Vary epsilon and disucss results
    print("Varying epsilon for SARSA")
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
    for epsilon in epsilons[::-1]:
        q, _, rewards_history_sarsa = train_sarsa(max_episodes=1000, epsilon=epsilon)
        batched_rewards = np.mean(np.array(rewards_history_sarsa).reshape(-1, 100), axis=1)
        print("Epsilon: ", epsilon)
        test_policy(q, show=True)
        plt.plot(batched_rewards, label=str(epsilon))

    plt.title("SARSA performance with varying ε")
    plt.xlabel("Batched Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    # plt.savefig('SARSA_epsilon_performance.png', bbox_inches='tight')
    plt.show()

    print("Varying epsilon for q-learning")
    for epsilon in epsilons[::-1]:
        _, e, rewards_history_qlearning = train_qlearning(max_episodes=1000, epsilon=epsilon)
        cap = ((int)(e / 100)) * 100  # round down to nearest 100
        rewards_history_qlearning = rewards_history_qlearning[:cap]
        batched_rewards = np.mean(np.array(rewards_history_qlearning).reshape(-1, 100), axis=1)
        print("Epsilon: ", epsilon)
        test_policy(q, show=True)
        plt.plot(batched_rewards, label=str(epsilon))

    plt.title("Q-learning performance with varying ε")
    plt.xlabel("Batched Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    # plt.savefig('ql_epsilon_performance.png', bbox_inches='tight')
    plt.show()

    print("Varying epsilon for q-learning with replay buffer")
    for epsilon in epsilons[::-1]:
        q, e, rewards_history_qlearning_replay = train_qlearning_replay(max_episodes=1000, epsilon=epsilon)
        cap = ((int)(e / 100)) * 100  # round down to nearest 100
        rewards_history_qlearning_replay = rewards_history_qlearning_replay[:cap]
        batched_rewards = np.mean(np.array(rewards_history_qlearning_replay).reshape(-1, 100), axis=1)
        print("Epsilon: ", epsilon)
        test_policy(q, show=True)  # Checking if the policy is optimal
        plt.plot(batched_rewards, label=str(epsilon))

    plt.title("Q-learning replay performance with varying ε")
    plt.xlabel("Batched Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.show()

def Q3():
    # Q3. Effect of snakepit
    print("Policy with snakepit")
    Q_sarsa, _, _ = train_sarsa(max_episodes=2500, epsilon=0.01, pit=True)
    print("Testing sarsa")
    test_policy(Q_sarsa, show=True, pit=True)
    print("")

    Q_qlearning, _, _ = train_qlearning(pit=True)
    print("Testing qlearning")
    test_policy(Q_qlearning, show=True, pit=True)
    print("")

    Q_qlearning_replay, _, _ = train_qlearning_replay(pit=True)
    print("Testing qlearning_replay")
    test_policy(Q_qlearning_replay, show=True, pit=True)
    print("")

def main():
    # Testing environment
    Q1()
    Q2()
    Q3()

if __name__ == "__main__":
    main()