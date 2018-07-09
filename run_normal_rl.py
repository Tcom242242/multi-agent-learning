import os, sys
sys.path.append(os.getcwd())
from games.game import Game
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import math
from agents.phc_agent import PHCAgent
from agents.policy import NormalPolicy
from games.simple_game import SimpleGame

if __name__ == '__main__':

    nb_agents = 2 
    agents = []
    for idx in range(nb_agents):
        policy = NormalPolicy()
        agent = PHCAgent(alpha=0.1, policy=policy, action_list=np.arange(2))  # agentの設定
        agents.append(agent)

    game = SimpleGame(nb_steps=400000, agents=agents)
    game.run()
    for idx, agent in enumerate(agents):
        print("agent{}s average reward:{}".format(idx, np.mean(agent.rewards)))
    plt.plot(np.arange(len(agents[0].pi_history)),agents[0].pi_history)
    plt.ylim(0, 1)
    plt.show()
