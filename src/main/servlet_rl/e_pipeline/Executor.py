import concurrent.futures
import copy
import os
from concurrent.futures import ProcessPoolExecutor

import torch

from src.main.servlet_rl.e_pipeline.mcts.MCTSPipeline import MCTSPipeline
from src.main.utils.Logger import logprint


class Executor:
    def __init__(self, pipeline_train: MCTSPipeline, pipeline_cpu: MCTSPipeline = None):
        # ___pipeline_______！
        self.pipeline_cpu = pipeline_cpu
        self.pipeline_train = pipeline_train

    def runAsync(self, iteration, iterationSaveModel, playoutN=1, trainN=10, samplePercentN=0.5, deviceName="cpu",
                 dirName=""):
        iteration //= iterationSaveModel
        if dirName != "":
            self.pipeline_train.load(dirName=dirName)
        for i in range(iteration):
            for j in range(iterationSaveModel):
                self.pipeline_train.evaluating()
                self.pipeline_train.collecting(playoutN=playoutN)
                self.pipeline_train.learning(trainN=trainN, samplePercentN=samplePercentN, deviceName=deviceName)
            self.pipeline_train.save(dirName=dirName)
        self.pipeline_train.evaluating()

    def runSync(self, iteration, iterationSaveModel, playoutN=1, trainN=10, samplePercentN=1, deviceName="cpu"):

        # _________
        deviceName = self.pipeline_train.agent.device
        iteration = iteration
        iterationSaveModel = iterationSaveModel
        playoutN = playoutN
        trainN = trainN
        samplePercentN = samplePercentN

        # 1.__env,modelManager__,____
        pipeline_cpu = self.pipeline_cpu
        pipeline_train = self.pipeline_train

        iteration //= iterationSaveModel

        # __ThreadPoolExecutor_________ProcessPoolExecutor
        # max_workers = os.cpu_count()- 2
        max_workers = os.cpu_count() // 4 + 1
        logprint.info(f"max_workers:{max_workers}")
        every_process_playoutN = playoutN // max_workers
        # every_process_playoutN = 1
        if every_process_playoutN < 1:
            every_process_playoutN = 1
        # __ multiprocessing ______ 'spawn'

        # multiprocessing.set_start_method('spawn', force=True)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            for i in range(iteration):
                for j in range(iterationSaveModel):
                    logprint.info(f"iteration:{i * iterationSaveModel + j},iterationSaveModel:{j}")

                    tasks = []

                    # __eval__

                    tasks.append(executor.submit(pipeline_cpu.evaluating))
                    # __collect__

                    for _ in range(max_workers):
                        tasks.append(executor.submit(pipeline_cpu.collecting, every_process_playoutN))
                    if deviceName == "cuda":
                        # cpu____,______,______
                        pipeline_train.learning(trainN=trainN, samplePercentN=samplePercentN, deviceName="cpu")

                    # _________
                    resultList = []

                    # __
                    for future in concurrent.futures.as_completed(tasks):
                        resultList.append(future.result())

                    # ==============================

                    logprint.info("1-mergeBuffer,self.pipeline_train.rlContext.bufferLen:{}".format(
                        len(pipeline_train.rlContext.buffer.rootNode_bestNodeList)))
                    for idx, result in enumerate(resultList):
                        try:
                            # _________

                            pipeline_train.rlContext.buffer.mergeBuffer(result)
                            pass

                        except Exception as e:
                            print(f"Task encountered an error: {e}")
                            # ___________________
                    logprint.info("2-mergeBuffer,self.pipeline.rlContext.bufferLen:{}".format(
                        len(pipeline_train.rlContext.buffer.rootNode_bestNodeList)))
                    logprint.info("1-self.pipeline_cpu.rlContext.bufferLen:{}".format(
                        len(pipeline_cpu.rlContext.buffer.rootNode_bestNodeList)))
                    # ________
                    count = 0
                    for task in tasks:
                        if not task.done():
                            count += 1
                    if deviceName == "cpu":
                        # cpu___,_________,______
                        pipeline_train.learning(trainN=trainN, samplePercentN=samplePercentN, deviceName="cpu")

                    pipeline_cpu.agent.model.load_state_dict(pipeline_train.agent.model.state_dict())
                    logprint.info(f"Task  is still running : count {count}")
                    # pipeline.load()
                pipeline_train.save()

        pipeline_train.evaluating(isSelfplayMode=True)


def verifyTrainEfficiency(pipeline: MCTSPipeline):
    representNet = copy.deepcopy(pipeline.agent.modelManager.representNet)
    predictNet = copy.deepcopy(pipeline.agent.modelManager.predictNet)
    # dynamicNet=copy.deepcopy(self.pipeline.agent.modelManager.dynamicNet)

    # context

    representNet2 = pipeline.agent.modelManager.representNet
    predictNet2 = pipeline.agent.modelManager.predictNet

    # dynamicNet2=self.pipeline.agent.modelManager.dynamicNet

    def is_same_model(model1, model2):
        # model1____
        model1 = model1  # _______
        model2 = model2  # _______
        params1 = {name: param for name, param in model1.named_parameters()}
        params2 = {name: param for name, param in model2.named_parameters()}
        averaged_params = {}
        is_same_model = True
        for name in params1.keys():
            assert name in params2, f"Missing parameter {name} in model2"
            assert params1[name].shape == params2[name].shape, f"Mismatched shapes for parameter {name}"
            #
            if not torch.equal(params1[name], params2[name]):
                is_same_model = False
            # if params1[name] is not params2[name]:
            #     is_same_model = False
            averaged_params[name] = (params1[name] + params2[name]) / 2.0
        # merged_model = model1  # ______，___model1_model2__，______________
        # for name, param in merged_model.named_parameters():
        #     assert name in averaged_params, f"Missing averaged parameter {name}"
        #     param.data.copy_(averaged_params[name])
        logprint.info(f"is_same_model: {is_same_model}")
        # return merged_model

    is_same_model(predictNet, predictNet2)
    is_same_model(predictNet, predictNet)
