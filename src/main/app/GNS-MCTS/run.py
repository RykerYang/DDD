import unittest

from src.main.servlet_rl.e_pipeline.Executor import Executor
from src.main.servlet_rl.e_pipeline.mcts.MCTSPipeline import MCTSPipeline
from src.main.servlet_rl.h_env.f_dgantt.g_Instance import generateRamdomInstance


class Main(unittest.TestCase):
    def test_runAsync(self, deviceName="cuda"):
        # 1.__env,modelManager__,____
        maxJobN = 30
        maxOpN = 15
        maxMacN = 30
        maxTime = 20
        minJobN = 1
        minOpN = 2
        minMacN = 1
        instance = generateRamdomInstance(maxJobN=maxJobN, maxOpN=maxOpN, maxMacN=maxMacN,
                                          maxTime=maxTime, minJobN=minJobN, minOpN=minOpN, minMacN=minMacN,
                                          random_seed=-1)
        start_code = instance.getRandomCode()
        envConfig = dict(
            instanceId=-1, start_codeId=-1, stopTimePercent=0, brokenMacId=-1, macRecoveryTime=0,
            instance=instance, start_code=start_code
        )
        # 2.pipeline
        pipeline = MCTSPipeline()
        pipeline.compileConfig(envConfig=envConfig,
                               simulationN=30, branchN=20, branchLen=10,
                               deviceName=deviceName)
        # 3.__executor,__run()
        executor = Executor(pipeline)
        executor.runAsync(iteration=2, iterationSaveModel=2, playoutN=1, trainN=3, samplePercentN=1,
                          deviceName=deviceName)

    def test_evaluate(self, deviceName="cuda"):
        # pipeline = MCTSPipeline()
        # pipeline.compileConfig(getEnvConfig=getEnvConfig, getModelManagerConfig=getModelManagerConfig,
        #                        simulationN=30, branchN=20, branchLen=5,
        #                        stopTimePercent=0,
        #                        brokenMacId=-1,
        #                        macRecoveryTime=0,
        #                        instanceId=1,
        #                        start_codeId=3, deviceName=deviceName)
        # 1.__env,modelManager__,____
        maxJobN = 80
        maxOpN = 10
        maxMacN = 80
        maxTime = 20
        minJobN = 79
        minOpN = 9
        minMacN = 79
        instance = generateRamdomInstance(maxJobN=maxJobN, maxOpN=maxOpN, maxMacN=maxMacN,
                                          maxTime=maxTime, minJobN=minJobN, minOpN=minOpN, minMacN=minMacN,
                                          random_seed=-1)
        start_code = instance.getRandomCode()
        envConfig = dict(
            instanceId=-1, start_codeId=-1, stopTimePercent=0, brokenMacId=-1, macRecoveryTime=0,
            instance=instance, start_code=start_code
        )
        # 2.pipeline
        pipeline = MCTSPipeline()
        pipeline.compileConfig(envConfig=envConfig,
                               simulationN=30, branchN=20, branchLen=10,
                               deviceName=deviceName)
        pipeline.evaluating(isSelfplayMode=True)

    def test_runSync(self, deviceName="cuda", simulationN=30, branchN=20, branchLen=5):
        # 1.__env,modelManager__,____
        # 1.__env,modelManager__,____
        maxJobN = 50
        maxOpN = 50
        maxMacN = 50
        maxTime = 50
        minJobN = 1
        minOpN = 2
        minMacN = 1
        instance = generateRamdomInstance(maxJobN=maxJobN, maxOpN=maxOpN, maxMacN=maxMacN,
                                          maxTime=maxTime, minJobN=minJobN, minOpN=minOpN, minMacN=minMacN,
                                          random_seed=-1)
        start_code = instance.getRandomCode()
        envConfig = dict(
            instanceId=-1, start_codeId=-1, stopTimePercent=0, brokenMacId=-1, macRecoveryTime=0,
            instance=instance, start_code=start_code
        )
        # 2.pipeline
        pipeline_cpu = MCTSPipeline()

        pipeline_cpu.compileConfig(envConfig=envConfig,
                               simulationN=30, branchN=20, branchLen=10,
                               deviceName="cpu")

        pipeline_train = MCTSPipeline()
        pipeline_train.compileConfig(envConfig=envConfig,
                               simulationN=30, branchN=20, branchLen=10,
                               deviceName=deviceName)
        # 2.__executor,__run()

        executor = Executor(pipeline_cpu=pipeline_cpu, pipeline_train=pipeline_train)
        # ____trainN=40_,____!!!trainN=2-5_____!
        executor.runSync(iteration=400, iterationSaveModel=10, playoutN=4, trainN=3, samplePercentN=1)

