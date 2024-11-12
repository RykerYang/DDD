from typing import List, Dict

import numpy as np

from src.main.servlet_rl.i_rlto.State import State
from src.main.utils.Logger import logprint


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    def __init__(self, parentNode: "TreeNode" = None, actionIdx=-1,
                 prob=0.0, qVal=0.0, needExplore: bool = False,
                 state: State = None,
                 c2: float = 1.25,
                 c1: float = 19652.0,
                 discount_factor: float = 0.96):
        # ______
        if not isinstance(parentNode, TreeNode) and parentNode is not None:
            raise TypeError("parentNode must be a TreeNode or None")
        if not isinstance(actionIdx, int) or actionIdx < -1:
            raise ValueError("actionIdx must be an integer >= -1")
        if prob < 0 or prob > 1:
            raise ValueError("prob must be a float between 0 and 1")
        if not isinstance(needExplore, bool):
            raise TypeError("needExplore must be a boolean")
        if not isinstance(c2, float) or c2 <= 0:
            raise ValueError("c2 must be a positive float")
        if not isinstance(c1, float) or c1 <= 0:
            raise ValueError("c1 must be a positive float")
        if not isinstance(discount_factor, float) or discount_factor <= 0:
            raise ValueError("discount_factor must be a positive float")
        assert not (parentNode is None and state is None), "____. parentNode_state______._________.__________"

        # ______
        self.parentNode: TreeNode = parentNode
        # pre_edge__
        self.actionIdx: int = actionIdx
        # πt _____
        self.mcts_prob: float = 0
        # pt
        self.prob: float = prob
        # zt
        self.mcts_qVal: float = 0.0000000000000001
        # vt
        self.qVal: float = qVal
        # ______，_env___！
        self.state: State = state
        #
        self.isChildRoot: bool = False
        # _____(actionIdx,node)
        self.childs: Dict[int, TreeNode] = {}
        # ____
        self.visit_count = 0
        # select info
        self.c2: float = c2
        self.c1: float = c1
        # ____
        self.needExplore: bool = needExplore
        self.discount_factor: float = discount_factor
        self.mu = 0.0  # ___ 0
        self.sigma = 0.05  # ____ 0.05

    def setExploreParam(self, needExplore: bool, mu: float = 0, sigma: float = 1):
        self.mu = mu  # ___ 0
        self.sigma = sigma  # ____ 0.05
        self.needExplore: bool = needExplore

    def move(self):
        """______"""

        # __+____
        # parent_mean_q: float = sum([node.mcts_qVal for node in self.childs.values()]) / len(self.childs)
        fitnessList: List[float] = []
        for node in self.childs.values():
            if node.state == None:
                fitnessList.append(float("inf"))
            else:
                fitnessList.append(node.state.fitnessDict["makespan"])
        try:
            bestAction = np.argmin(fitnessList)
        except Exception as e:
            logprint.warning(e)

        return bestAction, self.childs[bestAction]

    def select(self,noiseRate: float = 0.3):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """

        # if self.needExplore == False:
        #     # __+____
        #     # parent_mean_q: float = sum([node.mcts_qVal for node in self.childs.values()]) / len(self.childs)
        #     total_children_visit_counts: float = sum([node.visit_count for node in self.childs.values()])
        #     bestAction = max(self.childs.items(),
        #                      key=lambda act_node: act_node[1].getUcbScore(
        #                          # parent_mean_q=parent_mean_q,
        #                          total_children_visit_counts=total_children_visit_counts))[0]
        #
        # else:
        #     total_children_visit_counts: float = sum([node.visit_count for node in self.childs.values()])
        #     keys, vals = zip(*self.childs.items())
        #
        #     p_normal = np.random.normal(0, 1, len(self.childs))
        #     # p_normal = np.array([node.getUcbScore(total_children_visit_counts=total_children_visit_counts) for node in
        #     #                      self.childs.values()])
        #     p_normalized = (p_normal - p_normal.min()) / (p_normal.max() - p_normal.min())
        #     p_normalized = p_normalized / sum(p_normalized)
        #     p_normalized[0] += max(1.0 - sum(p_normalized), 0)
        #     # logprint.info("p_normalized:{}".format(p_normalized))
        #     # logprint.info("p_normalized[0]:{}".format(p_normalized[0]))
        #
        #     bestAction = np.random.choice(keys, p=p_normalized)
        #
        # return bestAction, self.childs[bestAction]
        assert noiseRate>=0 and noiseRate<=1, "noiseRate must be a float between 0 and 1"
        if self.needExplore == False:
            # __+____
            total_children_visit_counts: float = sum([node.visit_count for node in self.childs.values()])

            bestAction = max(self.childs.items(),
                             key=lambda act_node: act_node[1].getUcbScore(
                                 # parent_mean_q=parent_mean_q,
                                 total_children_visit_counts=total_children_visit_counts))[0]

            # _____
            # 2023_12_13_20:25:36, __makespan 92
            # 2023_12_14_9:28, __makespan 116,___,_rl__
            # bestAction = max(self.children.items(),
            #                  key=lambda act_node: act_node[1]._P)[0]
        else:
            temp = 1e-3
            act_visits = [(act, node.visit_count)
                          for act, node in self.childs.items()]
            acts, visits = zip(*act_visits)
            act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
            bestAction = np.random.choice(
                acts,
                # p=0.75 * act_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(act_probs)))
                p=(1-noiseRate) * act_probs + noiseRate * np.random.dirichlet(0.3 * np.ones(len(act_probs)))
            )
        return bestAction, self.childs[bestAction]

    def expand(self, probs: List[float], qVals: List[float]):
        """Expand tree by creating new children.
                action_priors: a list of tuples of actions and their prior probability
                    according to the policy function.
        """
        # if state is None and self.state is None:
        #     raise Exception("state is None")  # _____________None，_____
        # elif state is not None and self.state is None:
        #     self.state = state  # ________None______None，_______
        # elif state is not None and self.state is not None:
        #     print("state is not None")  # ______________None，_____

        child = {}
        for action, prob in enumerate(probs):
            if action not in child:
                child[action] = TreeNode(parentNode=self, actionIdx=action, state=None, prob=prob, qVal=qVals[action],
                                         needExplore=self.needExplore)

        self.childs: Dict[int:TreeNode] = child

    def backtrack(self) -> None:
        """
        Overview:
            Update the value sum and visit count of nodes along the search path.
        Arguments:
            - search_path: a vector of nodes on the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - to_play: which player to play the game in the current node.
            - value: the value to propagate along the search path.
            - discount_factor: the discount factor of reward.
        """

        # for play-with-bot-mode
        bootstrap_value = 0
        node = self
        done = False
        while not done:
            if node.isChildRoot:
                done = True
            node.mcts_qVal += self.qVal
            node.visit_count += 1

            true_reward = node.state.reward
            bootstrap_value = true_reward + self.discount_factor * bootstrap_value
            node = node.parentNode

        pass

    def isLeaf(self):
        if len(self.childs) == 0:
            return True
        else:
            return False

    # 2
    def getUcbScore(self, total_children_visit_counts: float, c_puct=100) -> float:
        """
        Overview:
            Compute the ucb score of the child.
            Arguments:
                - child: the child node to compute ucb score.
                - min_max_stats: a tool used to min-max normalize the score.
                - parent_mean_q: the mean q value of the parent node.
                - is_reset: whether the value prefix needs to be reset.
                - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
                - parent_value_prefix: the value prefix of parent node.
                - pb_c_base: constants c2 in muzero.
                - c1: constants c1 in muzero.
                - disount_factor: the discount factor of reward.
                - players: the number of players.
                - continuous_action_space: whether the action space is continuous in current env.
            Outputs:
                - ucb_value: the ucb score of the child.
        """
        # k1 = math.log((total_children_visit_counts + self.c2 + 1) / self.c2) + self.c1
        # k2 = (math.sqrt(total_children_visit_counts) / (self.visit_count + 1))
        # # ___+_______
        # ucb_score = k1 * k2 * self.prob + self.mcts_qVal

        u = c_puct * self.prob / (1 + self.visit_count)
        ucb_score = u + self.mcts_qVal

        return ucb_score
