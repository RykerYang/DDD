import os
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.main.servlet_rl.f_InteractOp.mcts.g_TreeNode import TreeNode
from src.main.servlet_rl.g_agent.ModelManager import ModelManager
from src.main.servlet_rl.i_rlto.State import State
from src.main.utils.Logger import logprint
from src.main.utils.Path import getShareResourcePath


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Agent:
    ''' _______SAC__ '''

    # agent_,_____________,__________!!!
    # __:1.________Agent_
    #     2. ___env_Agent_____

    def __init__(self, modelManagerConfig: Dict, lr=1e-4):
        self.modelManager: ModelManager = ModelManager(**modelManagerConfig)
        self.device: str = self.modelManager.deviceName
        # ____
        self.model: torch.nn.Module = self.modelManager.getModel(pqMode=3)
        # self.model.share_memory()
        self.action_probList: List
        # self.model.eval()
        # ____
        self.entropy_weight = 0.01
        self.value_weight = 0.1

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def takeActionProb(self, state: State) -> Dict[str, List[float]]:

        # todo _________eval__
        # device = self.device
        nodeList = state.nodeList
        edgeList = state.edgeList
        edgeTypeList = state.edgeTypeList
        actionTable = state.actionTable
        result = {}

        if len(edgeList) <= 0:
            action_priors = [1 for _ in range(len(nodeList) ** 2)]
            self.action_priors = list(softmax(np.array(action_priors)))
        else:

            # nodeTensor = torch.tensor(nodeList, dtype=torch.long)
            # edgeTensor = torch.LongTensor(np.array(edgeList).transpose()).to(device)
            # edgeTensor.view(2, len(edgeTypeList))
            # edgeTypeTensor = torch.LongTensor(edgeTypeList).to(device)
            # actionTensor = torch.tensor(actionList, dtype=torch.float).to(device)
            # actionTensor = actionTensor.view(1, -1, 1)
            nodeTensor = self.getNodeTensor(nodeList)
            edgeTensor = self.getEdgeTensor(edgeList)
            edgeTypeTensor = self.getEdgeTypeTensor(edgeTypeList)
            # actionTensor = self.getActionTensor(actionList)
            # model_output:List = []
            with torch.no_grad():
                # try:
                model_output = self.model(nodeTensor, edgeTensor, edgeTypeTensor, actionTable)
                # except Exception as e:
                #     logprint.warning(e)
                #     # import resource
                #     #
                #     # # ________（______）
                #     # available_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                #     #
                #     # # ____（_MB___）
                #     # available_memory_mb = available_memory / (1024.0 ** 2)
                #     #
                #     # logprint.info(f"Available memory: {available_memory_mb:.2f} MB")
                #
                #     logprint.info("error")

            probs = model_output[0]
            qvals = model_output[1]

            # __probs__，_____idx0
            result["probs"] = probs.tolist()
            result["qVals"] = qvals.tolist()
            self.action_probList = result["probs"]

        return result

    # def getAction_probList(self):
    #     # ______takeActionByPNet()_____(________)
    #     return self.action_probList

    def update(self, trainN, rootNode, bestNode, stepN=0):

        # ______________n_, ____________
        # =1. _bestNode_____rootNode, childNode__childNodeList
        childNodeList = []
        childNodeTemp: TreeNode = bestNode
        while childNodeTemp != rootNode:
            if childNodeTemp.isChildRoot == True:
                childNodeList.insert(0, childNodeTemp)
            childNodeTemp = childNodeTemp.parentNode
        # =2.
        try:
            assert len(childNodeList) - stepN > 0, "stepN is too large"
        except Exception as e:
            logprint.error(e)
            logprint.error(f"len(childNodeList) - stepN = {len(childNodeList) - stepN}")
            logprint.error(f"len(childNodeList) = {len(childNodeList)}")
            logprint.error(f"stepN = {stepN}")
            logprint.error(f"childNodeList = {childNodeList}")
        for childNodeIdx in range(len(childNodeList) - stepN):
            # node
            startNode = childNodeList[childNodeIdx]
            endNode = childNodeList[childNodeIdx + stepN]
            # mcts_____
            # __mcts_prob
            total_mcts_qVal = sum([child.mcts_qVal for child in endNode.childs.values()])
            for child in endNode.childs.values():
                child.mcts_prob = child.mcts_qVal / total_mcts_qVal
            mcts_probs = [node.mcts_prob for node in endNode.childs.values()]
            mcts_probs_tensor = torch.tensor(mcts_probs).to(self.device)
            mcts_qVals = [node.mcts_qVal for node in endNode.childs.values()]
            mcts_qVals_tensor = torch.tensor(mcts_qVals).to(self.device)
            # state
            nodeList = startNode.state.nodeList
            edgeList = startNode.state.edgeList
            edgeTypeList = startNode.state.edgeTypeList
            actionTable = startNode.state.actionTable
            # actionList
            actionList = []
            node = endNode
            while node != startNode:
                actionList.insert(0, node.actionIdx)
                node = node.parentNode
            # ____
            stepN = len(actionList)
            # stepN_actionList______,___stepN_actionTensor
            model = self.modelManager.getModel(pqMode=3, stepN=stepN)
            model.train()
            # ___

            # tensor
            nodeTensor = self.getNodeTensor(nodeList)
            edgeTensor = self.getEdgeTensor(edgeList)
            edgeTypeTensor = self.getEdgeTypeTensor(edgeTypeList)
            # action__, ____?actionTable
            # actionTensorList = self.getActionTensor(actionList)
            # actionObjList = [self.modelManager.actionTable[actionIdx] for actionIdx in actionList]
            # train
            for i in range(trainN):
                probs, qVals = model(nodeTensor, edgeTensor, edgeTypeTensor, actionTable)
                # ==========================================================
                log_probs = torch.log(probs)

                # calculate policy entropy, for monitoring only
                entropy = torch.mean(-torch.sum(probs * log_probs))
                entropy_loss = -entropy

                # ============
                # policy loss
                # ============
                policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_probs))

                # ============
                # value loss
                # ============
                value_loss = F.mse_loss(qVals, mcts_qVals_tensor)
                total_loss = self.value_weight * value_loss + policy_loss + self.entropy_weight * entropy_loss
                logprint.info(
                    f"value_loss:{value_loss.item()},policy_loss:{policy_loss.item()},entropy_loss:{entropy_loss.item()}")
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            # =====================______===========================
            # import torch.nn.utils as nn_utils
            #
            # # __________
            # params_and_grads = [(name, param, param.grad) for name, param in model.named_parameters() if
            #                     param.grad is not None]
            # ______
            # print("Gradient Information:")
            # for name, param, grad in params_and_grads:
            #     print(f"{name}:")
            #     if grad is not None:
            #         # __：________，____
            #         norm_grad = nn_utils.clip_grad_norm_(grad, max_norm=1., norm_type=2)
            #         print(f"  Gradient (L2 Norm): {norm_grad.item():.4f}")
            #         print(f"  Gradient (Unnormalized): {grad.detach().cpu().numpy()}")
            #
            #         # __：_________（__、____）
            #         mean, std = grad.mean().item(), grad.std().item()
            #         print(f"  Gradient Mean: {mean:.4f}, Std Deviation: {std:.4f}")
            #     else:
            #         print(f"  No gradient available for this parameter.")

    def load(self, headNetDirname="pq_head"):
        self.modelManager.loader(os.path.join(getShareResourcePath(), "model", headNetDirname), 0)
        self.modelManager.loader(os.path.join(getShareResourcePath(), "model", headNetDirname), 1)
        self.modelManager.loader(os.path.join(getShareResourcePath(), "model", headNetDirname), 2)
        self.modelManager.loader(os.path.join(getShareResourcePath(), "model", headNetDirname), 3)

    def save(self, headNetDirname="pq_head"):
        self.modelManager.saver(os.path.join(getShareResourcePath(), "model", headNetDirname), 0)
        self.modelManager.saver(os.path.join(getShareResourcePath(), "model", headNetDirname), 1)
        self.modelManager.saver(os.path.join(getShareResourcePath(), "model", headNetDirname), 2)
        self.modelManager.saver(os.path.join(getShareResourcePath(), "model", headNetDirname), 3)

    def getNodeTensor(self, nodeList: List[List[int]]) -> torch.Tensor:
        # __nodeList___nodeTensor
        nodeTensor = torch.tensor(nodeList, dtype=torch.int).to(self.device)
        return nodeTensor

    def getEdgeTensor(self, edgeList: List[List[int]]) -> torch.Tensor:
        # __edgeList___edgeTensor
        edgeTensor = torch.LongTensor(edgeList).to(self.device)
        return edgeTensor

    def getEdgeTypeTensor(self, edgeTypeList: List[List[int]]) -> torch.Tensor:
        # __edgeTypeList___edgeTypeTensor
        edgeTypeTensor = torch.LongTensor(edgeTypeList).to(self.device)
        return edgeTypeTensor

    def getActionTensor(self, actionList: List[int]) -> torch.Tensor:
        # __actionList___actionTensor
        actionTensor = torch.tensor(actionList, dtype=torch.float).to(self.device)
        actionTensor = actionTensor.view(1, -1, 1)
        return actionTensor

#
# def createSharedAgent(agent: Agent):
#     manager = multiprocessing.Manager()
#     Global = manager.Namespace()
#     Global.agent = agent
#     Global.agent.takeActionProb = agent.takeActionProb  # __step__
#
#     # return shared_agent
