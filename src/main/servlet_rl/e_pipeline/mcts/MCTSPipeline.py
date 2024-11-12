from typing import Dict, Any, List

from src.main.app.config import getModelManagerConfig
from src.main.servlet_rl.e_pipeline.IPipeline import IPipeline
from src.main.servlet_rl.f_InteractOp.mcts.e_MCTree import MCTree  # Replace with the actual path to the MCTree class
from src.main.servlet_rl.g_agent.Agent import Agent
from src.main.servlet_rl.h_env.e_dfjsp.e_Env import Env
from src.main.utils.Logger import logprint
from src.main.utils.Tensorboard import boardprint


class MCTSPipeline(IPipeline):
    def __init__(self):
        super().__init__()

        self.rlContext: MCTree = None
        self.simulationN: int
        self.branchN: int
        self.branchLen: int
        self.env: Env
        self.agent: Agent

    def compileConfig(self, envConfig: Dict[str, Any],
                      simulationN, branchN, branchLen, deviceName, afterCode: List[int] = None):
        """
        ______，__________。
        """
        # stopTimePercent = 0
        # brokenMacId = 0
        # macRecoveryTime = 0
        # instanceId = 1
        # start_codeId = 3
        self.simulationN = simulationN
        self.branchN = branchN
        self.branchLen = branchLen

        try:
            # ____
            env_config: Dict[str, Any] = envConfig
            # __: __getEnvConfig________，___________。

            env: Env = Env(**env_config)
            if afterCode is not None:
                env.reset(afterCode=afterCode)
            else:
                env.reset(afterCode=env.end_dgantt.start_gantt.start_code)

            # _______
            model_manager_config = getModelManagerConfig(macN=env.end_dgantt.start_gantt.instance.macN,
                                                         jobOpGN=env.end_dgantt.start_gantt.instance.jobOpGN,
                                                         actionTable=env.actionTable, deviceName=deviceName)

            # _____
            agent = Agent(modelManagerConfig=model_manager_config)
            # logprint.info(agent.model)

            # createSharedAgent(agent=agent)
            self.env = env
            self.agent = agent
            self.rlContext = MCTree(env=self.env, agent=self.agent)


        except Exception as e:
            # _______，______________
            logprint.warning(f"___________: {e}")
            # __________________________
            raise

    def collecting(self, playoutN=1):
        # collect playoutN_
        simulationN = self.simulationN
        branchN = self.branchN
        branchLen = self.branchLen

        assert self.rlContext != None, "___compileConfig()"
        makespan_list = []
        for i in range(playoutN):
            self.rlContext.interact(simulationN=simulationN, branchN=branchN, branchLen=branchLen, isSelfplayMode=True)
            logprint.info("makespan:{}".format(self.rlContext.bestNode.state.fitnessDict["makespan"]))
            makespan_list.append(self.rlContext.bestNode.state.fitnessDict["makespan"])
        boardprint.scalar("rl/collecting_makespan", {"average_makespan": sum(makespan_list) / len(makespan_list),
                                                     "max_makespan": max(makespan_list),
                                                     "min_makespan": min(makespan_list)})
        return self.rlContext.buffer
        # return True

    def learning(self, trainN=10, samplePercentN=0.5, deviceName="cpu"):
        # self.rlContext.agent.model.load_state_dict(self.rlContext.agent.model.state_dict())

        # TODO deviceName ___
        assert self.rlContext != None, "___compileConfig()"

        self.rlContext.train(trainN=trainN, samplePercentN=samplePercentN)
        self.rlContext.buffer.clear()

        # return self.rlContext.agent.model.state_dict()
        # return self.rlContext.agent.model

    def evaluating(self, isSelfplayMode=False):
        simulationN = self.simulationN
        branchN = self.branchN
        branchLen = self.branchLen
        assert self.rlContext != None, "___compileConfig()"

        self.rlContext.interact(simulationN=simulationN, branchN=branchN, branchLen=branchLen,
                                isSelfplayMode=isSelfplayMode)
        try:
            logprint.info("evaluating makespan explore-{}:{}".format(isSelfplayMode,
                                                                     self.rlContext.bestNode.state.fitnessDict[
                                                                         "makespan"]))
            boardprint.scalar("rl/makespan_explore-{}".format(isSelfplayMode),
                              self.rlContext.bestNode.state.fitnessDict["makespan"])
        except Exception as e:
            logprint.warning(f"_________: {e}")
        return self.rlContext.buffer

    def evaluating_endByTime(self, max_run_time, isSelfplayMode=False):
        simulationN = self.simulationN
        branchN = self.branchN
        branchLen = self.branchLen
        assert self.rlContext != None, "___compileConfig()"

        self.rlContext.interact_endByTime(max_run_time=max_run_time, branchN=branchN, branchLen=branchLen,
                                          isSelfplayMode=isSelfplayMode)
        try:
            logprint.info("evaluating makespan explore-{}:{}".format(isSelfplayMode,
                                                                     self.rlContext.bestNode.state.fitnessDict[
                                                                         "makespan"]))
        except Exception as e:
            logprint.warning(f"_________: {e}")
        return self.rlContext.buffer

    def save(self, dirName=""):
        if dirName == "":
            self.rlContext.agent.save()
        else:
            self.rlContext.agent.save(dirName)

    def load(self, dirName=""):
        if dirName == "":
            self.rlContext.agent.load()
        else:
            self.rlContext.agent.load(dirName)

    # def mergeModel(self, model1, model2):
    #     # model1____
    #     model1 = model1  # _______
    #     model2 = model2  # _______
    #     params1 = {name: param for name, param in model1.named_parameters()}
    #     params2 = {name: param for name, param in model2.named_parameters()}
    #     averaged_params = {}
    #     is_same_model = True
    #     for name in params1.keys():
    #         assert name in params2, f"Missing parameter {name} in model2"
    #         assert params1[name].shape == params2[name].shape, f"Mismatched shapes for parameter {name}"
    #         #
    #         # if not torch.equal(params1[name], params2[name]):
    #         #     is_same_model = False
    #         if params1[name] is not params2[name]:
    #             is_same_model = False
    #         averaged_params[name] = (params1[name] + params2[name]) / 2.0
    #     merged_model = model1  # ______，___model1_model2__，______________
    #     for name, param in merged_model.named_parameters():
    #         assert name in averaged_params, f"Missing averaged parameter {name}"
    #         param.data.copy_(averaged_params[name])
    #     logprint.info(f"is_same_model: {is_same_model}")
    #     return merged_model
