import datetime
import os
import pickle
import random
from typing import List, Tuple

from graphviz import Digraph

from src.main.servlet_rl.f_InteractOp.mcts.g_TreeNode import TreeNode
from src.main.utils.Logger import logprint
from src.main.utils.Path import getResourcePath


class MCTSBuffer:
    def __init__(self):
        self.rootNode_bestNodeList: List[Tuple[TreeNode, TreeNode]] = []
        pass

    def add(self, rootNode: TreeNode, bestNode: TreeNode):
        # 1.___buffer_
        self.rootNode_bestNodeList.append((rootNode, bestNode))

    def clear(self):
        self.rootNode_bestNodeList: List[Tuple[TreeNode, TreeNode]] = []

    def mergeBuffer(self, buffer: 'MCTSBuffer'):
        """
        ____MCTSBuffer______________。):
        """
        self.rootNode_bestNodeList.extend(buffer.rootNode_bestNodeList)

    def store(self, fileName=""):
        """
        _pickle ,_rootNode_bestNodeList__________。
        """
        if fileName == "":
            current_time = datetime.datetime.now()
            fileName = f"rootNode_bestNodeList_{current_time.strftime('%Y%m%d_%H%M%S')}.MCTSBuffer"
        filePath=os.path.join(getResourcePath(),fileName)
        with open(filePath, 'wb') as file:
            pickle.dump(self.rootNode_bestNodeList, file)
        return fileName

    def load(self, fileName):

        filePath=os.path.join(getResourcePath(),fileName)
        with open(filePath, 'rb') as file:
            self.rootNode_bestNodeList = pickle.load(file)


    def sample(self, samplePercentN):
        """
        _rootNode_bestNodeList___samplePercentN____。

        __:
        samplePercentN: _____，___0_1_____。

        __:
        _________。
        """
        # ____
        if not isinstance(samplePercentN, (int, float)) or not 0 <= samplePercentN <= 1:
            raise ValueError("samplePercentN___0_1______。")

        # ______
        rootNode_bestNodeList_length = len(self.rootNode_bestNodeList)
        if rootNode_bestNodeList_length == 0:
            # ______，_______
            return []

        # ______，_________
        sample_size = min(int(rootNode_bestNodeList_length * samplePercentN), rootNode_bestNodeList_length)

        # __random.sample______
        try:
            sample = random.sample(self.rootNode_bestNodeList, sample_size)
            if len(sample) == 0:
                return self.rootNode_bestNodeList
            return sample
        except ValueError as e:
            # __sample_size______，random.sample___ValueError__。___________。
            # __，________________。
            logprint.info(f"__：{e}，_______。")
            return self.rootNode_bestNodeList

    def getChildrenRootList(self, idx):
        # _____________，____rollout，__buffer
        pass

    def getPathFromRootToBestNode(self, idx) -> List[TreeNode]:
        # _______bestNode___
        pass

    def displayTree(self, idx):

        def visualize_tree(node: TreeNode, dot, parent=None, level=0):
            if node is None:
                return

            node_id = str(id(node))
            dot.node(node_id, str(node.state.fitnessDict.values()))

            if parent is not None:
                dot.edge(str(id(parent)), node_id)

            for child in node.childs.values():
                if child.state is None:
                    continue
                visualize_tree(child, dot, node, level + 1)

        # ____Graphviz__
        dot = Digraph(comment='Tree Visualization', format='png')

        # __________，_children_________
        root_node = self.rootNode_bestNodeList[idx][0]

        visualize_tree(root_node, dot)

        # _____________
        dot.render(view=True)  # _____PNG____________


