import sys
import os, sys
sys.path.append(os.getcwd())
from games.game import Game
import time
from tqdm import tqdm
import numpy as np
import ipdb
import math
import random
from agents.phc_agent import PHCAgent
from agents.policy import NormalPolicy
from games.simple_game import SimpleGame

if __name__ == '__main__':
    import random

    nb_agents = 2 
    agents = []
    for idx in range(nb_agents):
        policy = NormalPolicy()
        agent = PHCAgent(alpha=0.1, policy=policy, action_list=np.arange(2))  # agentの設定
        agents.append(agent)

    game = SimpleGame(nb_steps=100000, agents=agents)
    game.run()
