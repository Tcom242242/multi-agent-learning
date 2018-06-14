from envs.env import Env
import numpy as np
import ipdb
import ast

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

class GridEnv(Env):

    def __init__(self, section_conf, actions=[0, 1, 2, 3, 4], environment_changes=None):
        self.sections = self._init_sections(section_conf)
        self.actions = actions
        self.environment_changes = environment_changes

    def _init_sections(self, sections_conf):
        col = len(sections_conf)
        row = len(sections_conf[0])
        # sections = np.zeros((col, row))
        sections = [[0]*row]*col
        """
            confも二次元配列。それぞれの要素がhashで{reward :{}ここが変化していく}
        """
        for i in range(col):
            for j in range(row):
                sections[i][j] = Section(sections_conf[i][j])

        return sections

    def step(self, step, moves_info):
        log = {}
        agents_states = self.moves(moves_info)
        log = {"agents_states":agents_states}
        self.environment_change(step)
        return log

    def moves(self, moves_info):
        result = {}
        for move_info in moves_info:
            state = self.move(move_info["from_s"], move_info["action"])
            result[move_info["agent_id"]] = state
        return result

    def move(self, from_s, action):
        from_s = ast.literal_eval(from_s)
        to_x, to_y = from_s["x"], from_s["y"]

        # action = [-1, 0, 1]
        if action == UP:
            to_y += -1
        elif action == DOWN:
            to_y += 1
        elif action == LEFT:
            to_x += -1
        elif action == RIGHT:
            to_x += 1

        self.sections[from_s["y"]][from_s["x"]].nb_agents -= 1
        self.sections[to_y][to_x].nb_agents += 1
        to_s = {"x":to_x, "y":to_y}
        return to_s

    def force_move(self, from_s, to_s):
        from_s=ast.literal_eval(from_s)
        from_x, from_y = from_s["x"], from_s["y"]

        to_s=ast.literal_eval(to_s)
        to_x, to_y = to_s["x"], to_s["y"]

        self.sections[from_y][from_x].nb_agents -= 1
        self.sections[to_y][to_x].nb_agents += 1
        return to_s

    def environment_change(self, step):
        for section in self.sections:
            pass
            # section.size = self.environment_changes[str(section.id)][step]

    def add_agent_to_section(self, pos):
        self.sections[pos["y"]][pos["x"]].nb_agents += 1

    def step_log(self):
        for section in self.sections:
            section.reward = self.environment_changes[str(section.id)][step]

    def reset(self):
        pass

    def get_actions(self):
        return self.actions

    def get_config(self):
        pass

class Section():
    def __init__(self, conf):
        self.id = conf["id"]
        self.reward = conf["reward"]
        self.nb_agents = 0
