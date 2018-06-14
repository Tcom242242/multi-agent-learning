# from games.game import Game
# from game import Game
# from envs.bdp_env import BDPEnv
# from envs.rrsp_env import RRSPEnv
import sys
from games.game import Game
from agents.qlearning import QLearningAgent, RS_LUTA, QLearningAgentWithRSWoueV, \
    RS_CaP
from agents.policy import EpsGreedyQPolicy, EpsGreedyQPolicyVDBE, EpsGreedyQPolicyWoUE
import utils.reward_shaping_utils as rs

import time
from tqdm import tqdm
import numpy as np
import ipdb
import math


class SimpleGame(Game):
    """
        シンプルなゲーム理論で扱えるゲームを強化学習でやってみる 
    """
    def __init__(self, nb_eps=1, nb_steps=10000,reward_function_type="normal",environment_changes=None, 
                 reward_shaping_type=None,eta=None,sigma=None,conf_agents=None):
        self.conf_agents = conf_agents
        self.eta = eta
        self.sigma = sigma
        self.reward_matrix = self.create_reward_table(reward_function_type)
        self.reward_shape_type = reward_shaping_type
        self.agents = self._init_agents(conf_agents)
        self.nb_eps = nb_eps
        self.nb_steps = nb_steps
        self.update_interval = 20

    def _init_agents(self, conf_agents=None):
        if conf_agents is not None:
            return [self._init_agent(conf) for conf in conf_agents]
        else:
            for agent in self.agents:
                agent.init_state()
                agent.init_policy(policy)

    def _reset_agents(self):
        for agent in self.agents:
            from_s = agent.state
            to_s = agent.init_state()
            self.env.force_move(int(from_s), int(to_s))

    def _init_agent(self, conf):
        conf["epsilon_policy"] = None
        # conf["alpha"] = 0.0
        # conf["epsilon"] = 1.0
        conf["alpha_decay_rate"] = 0.99
        conf["epsilon_decay_rate"] = 0.99
        conf["epsilon_policy"] = "WoUE"
        if conf["epsilon_policy"] == "WoUE":
            # policy = EpsGreedyQPolicyWoUE(update_interval=100, eta=self.eta) 
            policy = EpsGreedyQPolicyWoUE(update_interval=200) 
        elif  conf["epsilon_policy"] =="VDBE":
            policy = EpsGreedyQPolicyVDBE(delta=(1/len(self.env.get_actions())), sigma=self.sigma) 
        else:
            policy = EpsGreedyQPolicy(conf["epsilon"])

        if conf["agent_type"] == "QLearning":
            agent = QLearningAgent(gamma=conf["gamma"], actions=[0, 1],
                                   observation=conf["ini_state"], alpha_decay_rate=conf["alpha_decay_rate"],
                                   epsilon_decay_rate=conf["epsilon_decay_rate"], q_values=None, id=conf["id"],
                                   name=None,
                                   alpha=conf["alpha"], training=True, policy=policy)
            # self.env.add_agent_to_section(conf["ini_state"])

        return agent

    def run(self):
        all_log = []
        for eps in range(self.nb_eps):
            exp_log = {"global_reward_history": [], "std": [],
                       "all_agents_average_reward_history": [], "all_agents_average_reward_std_history": [], 
                       "agents_log":[]}
            social_rewards = []
            for step in tqdm(range(self.nb_steps)):
                a0, a1 = self.agents[0].act(), self.agents[1].act()
                r0, r1 = self.reward_matrix[a0][a1]
                social_rewards.append(r0+r1)
                self.agents[0].get_reward(r0)
                self.agents[1].get_reward(r1)

                if self.agents[0].policy.name == "WoUE":
                    if (step+1) % self.agents[0].policy.update_interval == 0 and (step+1) > self.agents[0].policy.warmup:
                        mu = np.mean([agent.policy.sum_reward for agent in self.agents])
                        for agent in self.agents:
                            agent.update_eps(mu)

        social_rewards = np.array(social_rewards)
        print(self.agents[0].policy.eps_log.pop())
        print(np.mean(social_rewards))
        print("E:E0={}, E1={}".format(self.agents[0].q_values, self.agents[1].q_values))
        return None


    def create_reward_table(self, reward_function_type):
        """
            囚人のジレンマやチキン・ゲームなど、各ゲームに合わせて報酬行列を定義 
        """
        if reward_function_type == "normal":
            reward_matrix = [
                                [[6, 6], [2, 7]], 
                                [[7, 2], [0, 0]]
                            ]

        return reward_matrix


    def game_log(self):
        pass

    def get_conf(self):
        conf = {"agent_conf": agent_conf,
                "nb_eps": nb_eps,
                "nb_steps": nb_steps,
                "nb_sections": nb_sections,
                "sizes_sections": sizes_sections,
                "reward_function_type": self.reward_function_type,
                "reward_shape_type": self.reward_shape_type,
                "update_interval": self.update_interval}
        return conf

if __name__ == '__main__':
    import random

    nb_agents = 2 
    conf_agent = {"gamma": 0.1, "alpha": 0.1}
    conf_agents = [{"agent_type": "QLearning", "id": i, "gamma": 0.9,"epsilon":0.1, "alpha": 0.1, "ini_state": "s"}
                   for i in range(nb_agents)]
    game = SimpleGame(nb_steps=100000, conf_agents=conf_agents)
    game.run()
