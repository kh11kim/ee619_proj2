from env import Robot_Gridworld
from deep_q_learning import DeepQLearning
import matplotlib.pyplot as plt
import numpy as np
import torch

returns = []
gamma = 0.99


def print_q():
    x = np.linspace(-0.9, 0.9, 19, endpoint=True)
    y = np.linspace(-0.9, 0.9, 19, endpoint=True)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            state = torch.Tensor([x[i], y[j]])
            actions = dqn.policy_net(state).detach().numpy()
            Z[i, j] = np.min(actions)
    plt.matshow(Z)


def update():
    EPISODE = 300
    step = 0
    reward_per_episode = []
    for episode in range(EPISODE):

        state = env.reset()
        step_count = 0
        returns = []
        while True:
            env.render()
            action = dqn.choose_action(state, step)
            next_state, reward, terminal = env.step(action)

            step_count += 1
            dqn.store_transition(state, action, reward, next_state, terminal)

            if (step > 200) and (step % 5 == 0):
                dqn.learn()
            #### Begin learning after accumulating certain amount of memory #####
            state = next_state

            returns.append(reward)
            if terminal == True:
                print(" {} End. Total steps : {} avg_rewards : {}\n".format(
                    episode + 1, step_count, np.mean(returns)))
                break
            step += 1
        reward_per_episode.append(np.mean(returns))

   ############# Implement the codes to plot 'returns per episode' ####################
   ############# You don't need to place your plotting code right here ################
    print('Game over.\n')
    print_q()

    plt.plot(range(1, EPISODE+1), reward_per_episode, label="DQN")
    plt.xlabel("Episodes")
    plt.ylabel("Return per Episode")
    plt.legend()
    plt.savefig("learning_curve.pdf")
    env.destroy()


if __name__ == "__main__":

    env = Robot_Gridworld()

    ###### Recommended hyper-parameters. You can change if you want ###############
    dqn = DeepQLearning(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        discount_factor=0.95,
                        # e_greedy=0.1,
                        eps_start=0.9,
                        eps_end=0.05,
                        eps_decay=8000,
                        replace_target_iter=10,
                        memory_size=2000
                        )

    env.after(100, update)  # Basic module in tkinter
    env.mainloop()  # Basic module in tkinter
