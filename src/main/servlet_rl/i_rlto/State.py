from typing import List, Dict

from src.main.servlet_rl.i_rlto.Action import Action


class State:
    def __init__(self, nodeList: List[List[int]],
                 edgeList: List[List[int]],
                 edgeTypeList: List[List[int]],
                 afterCode: List[int],
                 fitnessDict: Dict[str, float],
                 reward: float):
        # ot
        # 	afterCode
        # 	fitness
        self.afterCode: List[int] = afterCode
        self.fitnessDict: Dict[str, float] = fitnessDict
        self.reward: float = reward
        # st
        # 	node
        # 	edge
        # 	edgeType
        self.nodeList: List[List[int]] = nodeList
        self.edgeList: List[List[int]] = edgeList
        self.edgeTypeList: List[List[int]] = edgeTypeList
        self.actionTable:  List[Action]= []
