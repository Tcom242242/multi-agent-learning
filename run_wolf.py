import os, sys
sys.path.append(os.getcwd())
from games.game import Game
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ipdb
# from agents.phc_agent import PHCAgent
from agents.wolf_agent import WoLFAgent
from agents.policy import NormalPolicy
from games.simple_game import SimpleGame

if __name__ == '__main__':

    nb_agents = 2 
    agents = []
    for idx in range(nb_agents):
        policy = NormalPolicy()
        agent = WoLFAgent(alpha=0.1, policy=policy, action_list=np.arange(3))  # agentの設定
        agents.append(agent)

    game = SimpleGame(nb_steps=100000, agents=agents)
    game.run()
    for idx, agent in enumerate(agents):
        print("agent{}s average reward:{}".format(idx, np.mean(agent.rewards)))
    # plt.plot(np.arange(len(agents[0].pi_history)),agents[0].pi_history)
    plt.plot(agents[0].pi_history, agents[1].pi_history)
    plt.ylabel("Agent1's Probability of selecting Paper")
    plt.xlabel("Agent2's Probability of selecting Paper")
    # plt.xlabel("Step")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
