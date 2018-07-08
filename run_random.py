import os, sys
sys.path.append(os.getcwd())
from games.game import Game
import time
from tqdm import tqdm
import numpy as np
from agents.random_agent import RandomAgent
from games.simple_game import SimpleGame

if __name__ == '__main__':
    nb_agents = 2 
    agents = []

    for idx in range(nb_agents):
        agent = RandomAgent(action_list=np.arange(2))
        agents.append(agent)

    game = SimpleGame(nb_steps=100000, agents=agents)
    game.run()
    # print results
    for idx, agent in enumerate(agents):
        print("agent{}s average reward:{}".format(idx, np.mean(agent.rewards)))

