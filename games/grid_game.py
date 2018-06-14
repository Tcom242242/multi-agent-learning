# from games.game import Game
from games.game import Game
from envs.grid_env import GridEnv
from agents.qlearning import QLearningAgent, RS_LUTA, QLearningAgentWithRSWoueV, \
    RS_CaP
from agents.policy import EpsGreedyQPolicy, EpsGreedyQPolicyVDBE, EpsGreedyQPolicyWoUE, EpsGreedyQPolicyWoUENewton
import utils.reward_shaping_utils as rs

import time
from tqdm import tqdm
import numpy as np
import ipdb
import math
import ast


STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

class GridGame(Game):
    """
        grid game 
        Input : nb_eps
    """
    def __init__(self, nb_eps=1, nb_steps=50, conf_agents=None, goal_conf=None):

        self.env = GridEnv(section_conf=section_conf, environment_changes=environment_changes)
        self.conf_agents = conf_agents
        self.agents = self._init_agents(conf_agents)
        self.nb_eps = nb_eps
        self.goal_conf = goal_conf
        self.goal_or_nots = [False for agent in self.agents]

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
            self.env.force_move(from_s, to_s)

    def _init_agent(self, conf):
        if conf["agent_type"] == "QLearning":
            agent = QLearningAgent(gamma=conf["gamma"], actions=self.env.get_actions(),
                                   observation=conf["ini_state"], alpha_decay_rate=conf["alpha_decay_rate"],
                                   epsilon_decay_rate=conf["epsilon_decay_rate"], q_values=None, id=conf["id"],
                                   name=None,
                                   alpha=conf["alpha"], training=True, policy=policy)
            self.env.add_agent_to_section(conf["ini_state"])
        return agent

    def is_allagent_goal(self):
        for i, agent in enumerate(self.agents):
            if self.is_agent_goal(agent) == False:
                return False
        return True

    def is_agent_goal(self, agent):
        state = ast.literal_eval(agent.state)
        x, y = state["x"], state["y"]
        gx, gy = self.goal_conf[agent.id]["x"], self.goal_conf[agent.id]["y"]
        if x==gx and y==gy:
            return True
        else:
            return False

    def run(self):
        all_log = {"step":[]}
        for eps in range(self.nb_eps):
            step = 0
            while True:
                if self.is_allagent_goal():
                    break

                moves_info = []
                for agent in self.agents:
                    if self.is_agent_goal(agent):
                        continue
                    from_s = agent.state
                    action = agent.act()
                    while (self.is_possible_action(action, from_s) == False):
                        action = agent.act()

                    move_info = {"agent_id": agent.id,
                                 "from_s": from_s,
                                 "action": action}
                    moves_info.append(move_info)

                log = self.env.step(step, moves_info)

                # compute_global_reward
                global_reward = self.compute_global_reward()
                all_agents_reward = []
                for agent in self.agents:
                    if self.is_agent_goal(agent):
                        continue
                    next_state = log["agents_states"][agent.id]
                    agent.observe(next_state)
                    reward = self.compute_local_reward(agent.id, next_state)
                    all_agents_reward.append(reward)    # 全エージェントの平均を計算する用
                step += 1

            # パラメータの更新
            for agent in self.agents:
                agent.decay_alpha()
                agent.decay_epsilon()

            self._reset_agents()
            all_log["step"].append(step)
        return all_log, self.agents, self.env

    def is_possible_action(self, action, state):
        """ 
            実行可能な行動かどうかの判定
        """
        state = ast.literal_eval(state)
        to_x = state["x"]
        to_y = state["y"]

        if action == STAY:
            return True
        elif action == UP:
            to_y += -1
        elif action == DOWN:
            to_y += 1
        elif action == LEFT:
            to_x += -1
        elif action == RIGHT:
            to_x += 1
        else:
            raize("Action Eroor")

        if len(self.env.sections) <= to_y or 0 > to_y:
            return False
        elif len(self.env.sections[0]) <= to_x or 0 > to_x:
            return False

        return True

    def compute_local_reward(self, agent_id, state):
        x = state["x"]
        y = state["y"]
        if self.goal_conf[agent_id]["x"] == x and self.goal_conf[agent_id]["y"] == y:
            reward = 100
        else:
            reward = -(self.env.sections[y][x].reward + (0.02*self.env.sections[y][x].nb_agents))
        return reward

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
    # todo
    # Q agentの初期化
    nb_agents = 10
    conf_agent = {"gamma": 0.9, "alpha": 0.9}
    conf_agents = [{"agent_type": "QLearning", "id": i, "gamma": 0.9, "alpha": 0.9, "ini_state": random.randint(0, 2)}
                   for i in range(nb_agents)]
    game = BDPGame(conf_agents=conf_agents)
