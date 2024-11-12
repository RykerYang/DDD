import copy
import time
from typing import Tuple, List

from src.main.servlet_rl.f_InteractOp.mcts.f_ChildRoot import ChildRoot
from src.main.servlet_rl.f_InteractOp.mcts.f_MCTSBuffer import MCTSBuffer
from src.main.servlet_rl.f_InteractOp.mcts.g_TreeNode import TreeNode
from src.main.servlet_rl.g_agent.Agent import Agent
from src.main.servlet_rl.h_env.e_dfjsp.e_Env import Env


class MCTree:
    def __init__(self, env: Env, agent: Agent):
        """
        __
            ___________
                __
                    mcts
                __
                    _______________
        agent env interactOp：____，_____！
            buffer_____interactOp___
                ______，____，__buffer
        trainer____agent，______interactOp?
            agent______，______，__agent____modelManager
        """
        assert env.curState is not None, "env ____reset____"
        # __________(_________________)
        self.buffer: MCTSBuffer = MCTSBuffer()
        # _________,_________
        self.env: Env = env

        self.agent: Agent = agent
        # ________
        # =2.create rootNode
        rootNode = TreeNode(parentNode=None, actionIdx=-1,
                            prob=0, needExplore=False, state=env.curState)
        # rootNode_backup = copy.copy(rootNode)
        rootNode.isChildRoot = True
        self.rootNode: TreeNode=rootNode
        self.bestNode: TreeNode
        # ________

    def interact(self, simulationN: int, branchN: int, branchLen: int, isSelfplayMode=False,max_run_time=-1):


        simulationN = int(simulationN)
        branchN: int = int(branchN)
        branchLen: int = int(branchLen)
        isSelfplayMode: bool = bool(isSelfplayMode)
        # =1.__childRoot
        env = self.env
        agent = self.agent
        agent.model.eval()
        childRoot = ChildRoot(env=env, agent=agent)
        # =2.Init root node
        rootNode=copy.deepcopy(self.rootNode)
        rootNode.setExploreParam(needExplore=isSelfplayMode, mu=0.0, sigma=0.05)


        # =3.mcts
        bestNode = rootNode

        if len(rootNode.state.actionTable)==0:
            self.bestNode = bestNode
            self.buffer.add(rootNode, bestNode)
            return
        for i in range(simulationN):
            # logprint.info(f"simulationN:{i}")
            bestNode.isChildRoot = True
            bestNode = childRoot.getNextChildrenRootNode(childRootNode=bestNode, branchN=branchN, branchLen=branchLen)


        self.bestNode = bestNode
        # =4.______buffer
        self.buffer.add(rootNode, bestNode)

    def interact_endByTime(self, max_run_time: int, branchN: int, branchLen: int, isSelfplayMode=False):



        branchN: int = int(branchN)
        branchLen: int = int(branchLen)
        isSelfplayMode: bool = bool(isSelfplayMode)
        # =1.__childRoot
        env = self.env
        agent = self.agent
        agent.model.eval()
        childRoot = ChildRoot(env=env, agent=agent)
        # =2.Init root node
        rootNode=copy.deepcopy(self.rootNode)
        rootNode.setExploreParam(needExplore=isSelfplayMode, mu=0.0, sigma=0.05)


        # =3.mcts
        bestNode = rootNode

        if len(rootNode.state.actionTable)==0:
            self.bestNode = bestNode
            self.buffer.add(rootNode, bestNode)
            return
        start_time = time.time()
        while True:
            cpu_time = time.time() - start_time
            if cpu_time > max_run_time:
                break
            # logprint.info(f"simulationN:{i}")
            bestNode.isChildRoot = True
            bestNode = childRoot.getNextChildrenRootNode(childRootNode=bestNode, branchN=branchN, branchLen=branchLen)


        self.bestNode = bestNode
        # =4.______buffer
        self.buffer.add(rootNode, bestNode)




    def train(self, trainN, samplePercentN):
        # trainN: ___________
        rootNode_bestNodeList: List[Tuple[TreeNode, TreeNode]] = self.buffer.sample(samplePercentN)
        for rootNode, bestNode in rootNode_bestNodeList:
            self.agent.update(trainN, rootNode, bestNode)
