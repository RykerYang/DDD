from src.main.servlet_rl.f_InteractOp.mcts.g_TreeNode import TreeNode
from src.main.servlet_rl.h_env.e_dfjsp import Env
from src.main.utils.Logger import logprint


class ChildRoot:
    def __init__(self, env, agent):
        self.env:Env = env
        # self.childRoot : TreeNode
        self.agent = agent

    def getNextChildrenRootNode(self, childRootNode: TreeNode, branchN, branchLen, needCrossNode=False):
        # gc.collect()  # ________
        try:
            assert childRootNode.isChildRoot and childRootNode.state is not None, "_____state___None"
        except AssertionError:
            logprint.info("_____state___None")
        bestNode = childRootNode

        for i in range(branchN):
            node = childRootNode
            # =3.select node
            branchCount = 0
            while not node.isLeaf():
                if branchCount >= branchLen:
                    break
                action, node = node.select()
                branchCount += 1

            # while branchCount <= branchLen:
            #
            #     # =4.evaluate node
            #     probs, qVals = self.evaluate(node)
            #     # =5.expand node
            #     node.expand(probs=probs, qVals=qVals)
            #     # =6.backtrack
            #     node.backtrack()
            #     branchCount += 1
            #
            #     # =7.update best node
            #     if node.state.reward > bestNode.state.reward:
            #         bestNode = node
            #     action, node = node.select()

            # =4.evaluate node
            probs, qVals = self.evaluate(node)
            # =5.expand node
            node.expand(probs=probs, qVals=qVals)
            # =6.backtrack
            node.backtrack()
            branchCount += 1

            # =7.update best node
            if node.state.reward > bestNode.state.reward:
                bestNode = node
        if needCrossNode == False:
            # ____,________
            action, bestNode = childRootNode.move()
        if childRootNode.needExplore:

            try:
                assert bestNode.state is not None, "_____state___None"
            except AssertionError:
                logprint.warning("_____state___None")
        return bestNode

    def getNextChildrenRootNode_train(self, childRootNode: TreeNode, branchN, branchLen):
        # gc.collect()  # ________
        try:
            assert childRootNode.isChildRoot and childRootNode.state is not None, "_____state___None"
        except AssertionError:
            logprint.info("_____state___None")
        bestNode = childRootNode

        for i in range(branchN):
            node = childRootNode
            # =3.select node
            branchCount = 0
            while not node.isLeaf():
                if branchCount >= branchLen:
                    break
                action, node = node.select()
                branchCount += 1

            while branchCount <= branchLen:

                # =4.evaluate node
                probs, qVals = self.evaluate(node)
                # =5.expand node
                node.expand(probs=probs, qVals=qVals)
                # =6.backtrack
                node.backtrack()

                branchCount += 1

                # =7.update best node
                if node.state.reward > bestNode.state.reward:
                    bestNode = node
                action, node = node.select()

        return bestNode

    def evaluate(self, node):
        # =1.init env agent
        agent = self.agent
        env = self.env
        # _____None_，____，_______.(_______)
        if node.state is None or len(node.state.nodeList)==0:
            # _______________
            try:
                env.reset(node.parentNode.state.afterCode)
            except :
                logprint.warning("______None")
            # _________，_______
            # TODO ____step_jobSeq
            # state = env.step_jobSeq(node.actionIdx, needDisplay=0, preState_fitnessDict=node.parentNode.state.fitnessDict)
            state = env.step(node.actionIdx, needDisplay=0, preState_fitnessDict=node.parentNode.state.fitnessDict)
            node.state = state

        action_result = agent.takeActionProb(node.state)

        probs = action_result["probs"]
        qVals = action_result["qVals"]
        return probs, qVals
