import copy
import gc
import random
from typing import List, Dict

import psutil

from src.main.servlet_rl.h_env.e_dfjsp.f_DataBase import DataBase as DataBaseEnv
from src.main.servlet_rl.h_env.f_dgantt.e_DGantt import DGantt
from src.main.servlet_rl.h_env.f_dgantt.f_Gantt import Gantt
from src.main.servlet_rl.h_env.f_dgantt.g_Instance import Instance
from src.main.servlet_rl.i_rlto.Action import Action
from src.main.servlet_rl.i_rlto.State import State
from src.main.utils.Logger import logprint
from src.main.utils.Path import getShareResourcePath


class Env:
    # 1.
    def __init__(self, stopTimePercent, brokenMacId, macRecoveryTime,
                 # ___
                 shareResourcePath: str = getShareResourcePath(),
                 rewardConfig: Dict[str, float] = {"makespan": 1.0},
                 maxConstant: int = 1000,
                 random_time_seed_dynamic_event=-1,
                 # ________,________
                 # option 1
                 instanceId=-1, start_codeId=-1,
                 # option 2
                 instance: Instance = None,
                 start_code: List[int] = None, needChangeActionTable=1):
        """

        Args:
            instanceId:
            start_codeId:
            stopTimePercent:
            brokenMacId:
            macRecoveryTime:
            rewardConfig: __fitness,____reward___. 0<=__<=1
            jobOpList:List[List[List[int]]]=None
            start_code:List[int]=None
        """
        # ________

        if instanceId == -1 or start_codeId == -1:
            if instance is None or start_code is None:
                raise ValueError("__________")
        else:
            # ________
            dataBaseEnv = DataBaseEnv(instanceId=instanceId, shareResourcePath=shareResourcePath)
            # ___________________
            jobOpList, start_code = dataBaseEnv.query_instance(start_codeId=start_codeId)
            # ______________
            instance: Instance = Instance(jobOpList=jobOpList)

        self.maxConstant = maxConstant

        # _______________
        start_gantt: Gantt = Gantt(start_code=start_code,
                                   instance=instance)
        # _________，_________、___MacId_Mac____
        # before_dgantt_clone,______________DGantt
        # ______before_dgantt,__________
        before_dgantt: DGantt = DGantt(start_gantt=start_gantt,
                                       stopTimePercent=stopTimePercent,
                                       brokenMacId=brokenMacId,
                                       macRecoveryTime=macRecoveryTime,
                                       maxConstant=maxConstant,
                                       random_time_seed_dynamic_event=random_time_seed_dynamic_event)

        # _______________
        self.end_dgantt = before_dgantt
        self.rewardConfig = rewardConfig

        # _______、___、___________
        self.curState: State = None
        self.nodeList: List
        self.edgeList: List
        self.edgeTypeList: List
        self.afterCode: List[int]

        # __Mac__
        self.macN = self.end_dgantt.start_gantt.instance.macN
        # ___________
        self.reward: int = 0
        self.fitnessDict: Dict[str, float] = dict()

        # _______________，____stoptime
        # initialActionTable________,_____afterCode__
        self.initialActionTable: List[Action] = []
        sorted_jobOpKey_remainNode = sorted(self.end_dgantt.jobOpKey_remainNode)
        for nodeIdxA, keyA in enumerate(sorted_jobOpKey_remainNode):
            for nodeIdxB, keyB in enumerate(sorted_jobOpKey_remainNode):
                if nodeIdxA >= nodeIdxB:
                    # ____
                    continue
                # __action_______,________,___!
                nodeA = self.end_dgantt.jobOpKey_remainNode[keyA]
                nodeB = self.end_dgantt.jobOpKey_remainNode[keyB]
                if nodeA.macId != nodeB.macId:
                    # ________
                    continue
                if (nodeA.execId - nodeB.execId) ** 2 > 1:
                    # ___
                    continue
                self.initialActionTable.append(Action(
                    actionIdx=len(self.initialActionTable) - 1,
                    nodeIdxA=nodeIdxA,
                    jobIdA=self.end_dgantt.jobOpKey_remainNode[keyA].jobId,
                    opIdA=self.end_dgantt.jobOpKey_remainNode[keyA].opId,
                    nodeIdxB=nodeIdxB,
                    jobIdB=self.end_dgantt.jobOpKey_remainNode[keyB].jobId,
                    opIdB=self.end_dgantt.jobOpKey_remainNode[keyB].opId))
        self.needChangeActionTable = needChangeActionTable
        self.actionTable: List[Action] = self.changeActionTableAccordingToAfterCode(
            needChangeActionTable=self.needChangeActionTable)

    def reset(self, afterCode: List[int], needCloneDgantt=1) -> State:
        """

        Args:
            afterCode: _______, _________
            needCloneDgantt:

        Returns:

        """
        assert afterCode != None, "afterCode is None"
        if len(afterCode) // 2 != len(self.end_dgantt.jobOpKey_remainNode):
            # todo logprint.warning("afterCode is not right")
            afterCode = self.end_dgantt.fixAfterCode(afterCode=afterCode)
            assert len(afterCode) // 2 == len(self.end_dgantt.jobOpKey_remainNode.keys()), "fix afterCode is not right"

        # 0.___nodeList,edgeList,edgeTypeList
        self.nodeList: List = []
        self.nodeBinList: List = []
        self.edgeList: List = []
        self.edgeTypeList: List = []
        self.afterCode = afterCode
        # self.discountVal = discountVal
        # 1.__afterCode__gantt
        jobIdCodeLen = self.end_dgantt.remainNodeN
        # if len(afterCode) // 2 != jobIdCodeLen:
        #     afterCode = self.end_dgantt.fixAfterCode(afterCode=afterCode)
        if needCloneDgantt == 1:
            self.end_dgantt = self.end_dgantt.clone()
        for idx in range(jobIdCodeLen):
            # 2.1 _jobId_macId
            jobId = afterCode[idx]
            macId = afterCode[idx + jobIdCodeLen]
            try:
                self.end_dgantt.appendNodeToEndGantt(jobId=jobId, macId=macId, afterCodeIdx=idx)
            except Exception as e:
                logprint.warning(e)
                raise ValueError()
        self.setFitnessDict()
        # self.setReward()
        self.setNodeEdgeState()

        # nodeBinList = self.setNodeBin(self.nodeList)
        # self.nodeBinList = nodeBinList
        # return nodeBinList, self.edgeList, self.edgeTypeList, self.afterCode
        # return self.nodeList, self.edgeList, self.edgeTypeList, self.afterCode

        state = State(nodeList=self.nodeList, edgeList=self.edgeList, edgeTypeList=self.edgeTypeList,
                      afterCode=self.afterCode, fitnessDict=self.fitnessDict, reward=self.reward)
        self.curState = state
        self.actionTable: List[Action] = self.changeActionTableAccordingToAfterCode(
            needChangeActionTable=self.needChangeActionTable)
        state.actionTable = self.actionTable

        return state

    def step(self, actionIdx: int, preState_fitnessDict=None, needDisplay=0) -> State:
        """

        Args:
            actionIdx: s_k___
            preState_fitnessDict: s_k_fitness
            needDisplay:

        Returns:

        """

        # 0.____local Search
        # 1.__cur_dgantt_code,A_B_jobIdCode__idx
        jobIdA: int = self.actionTable[actionIdx].jobIdA
        opIdA: int = self.actionTable[actionIdx].opIdA
        jobIdB: int = self.actionTable[actionIdx].jobIdB
        opIdB: int = self.actionTable[actionIdx].opIdB
        afterCode = self.afterCode
        jobCodeLen = len(afterCode) // 2
        # (jobId,macId)_____,___afterCodeIdx
        afterCodeIdx1 = self.end_dgantt.end_gantt.jobList[jobIdA][opIdA].afterCodeIdx
        afterCodeIdx2 = self.end_dgantt.end_gantt.jobList[jobIdB][opIdB].afterCodeIdx

        # 2.____lsCodeList

        # 2.1.____
        lsAfterCodeList = [copy.copy(self.afterCode) for _ in range(2)]
        lsAfterCodeList[1][afterCodeIdx1] = jobIdB
        lsAfterCodeList[1][afterCodeIdx2] = jobIdA

        lsAfterCodeList[0][afterCodeIdx1 + jobCodeLen] = -1
        lsAfterCodeList[0][afterCodeIdx2 + jobCodeLen] = -1
        lsAfterCodeList[1][afterCodeIdx1 + jobCodeLen] = -1
        lsAfterCodeList[1][afterCodeIdx2 + jobCodeLen] = -1

        # 2.____
        # dganttList
        dganttList = []
        # dganttList.append(self.end_dgantt)
        # _____(__,_____,__________jobIdOpId____)
        jobOpKey_macId__Dict = {}
        jobId_opIdInGanttFront = copy.copy(self.end_dgantt.jobId_opIdInGanttFront)
        for codeIdx in range(0, jobCodeLen):
            jobId = lsAfterCodeList[0][codeIdx]
            opId = jobId_opIdInGanttFront[jobId] + 1
            macId = lsAfterCodeList[0][codeIdx + jobCodeLen]
            jobId_opIdInGanttFront[jobId] += 1
            jobOpKey = jobId * self.maxConstant + opId
            jobOpKey_macId__Dict[jobOpKey] = macId
        # _______
        for idx, code in enumerate(lsAfterCodeList):
            # __code__
            macTime = [0 for _ in range(self.macN)]
            dgantt = self.end_dgantt.clone()
            for codeIdx in range(jobCodeLen):
                jobId = code[codeIdx]
                opId = dgantt.jobId_opIdInGanttFront_current[jobId] + 1
                jobOpKey = jobId * self.maxConstant + opId
                macId = jobOpKey_macId__Dict[jobOpKey]

                if macId == -1:
                    macIdTotalTimeDict = copy.copy(dgantt.end_gantt.jobList[jobId][opId].macIdTimeDict)
                    for macId, totalTime in enumerate(macTime):
                        if macId in macIdTotalTimeDict.keys():
                            macIdTotalTimeDict[macId] += totalTime
                    bestMacId, time = min(macIdTotalTimeDict.items(), key=lambda x: x[1])
                    code[codeIdx + jobCodeLen] = bestMacId
                    macId = bestMacId

                dgantt.appendNodeToEndGantt(jobId=jobId,
                                            macId=macId, afterCodeIdx=codeIdx)
                opId = dgantt.jobId_opIdInGanttFront_current[jobId]
                macTime[macId] += dgantt.end_gantt.jobList[jobId][opId].time

            dgantt.getDFitness()
            if needDisplay == 1:
                dgantt.displayDGantt()
                logprint.info(dgantt.end_gantt.fitness)
            dganttList.append(dgantt)

        # 3.__dganttList___
        # TODO ____reward
        bestDgantt = min(dganttList, key=lambda x: x.end_gantt.fitness)
        self.end_dgantt = bestDgantt
        # 4.__dgantt,_____
        self.setNodeEdgeState()
        self.setAfterCode()
        self.setFitnessDict()
        self.setReward(preState_fitnessDict)
        state = State(nodeList=self.nodeList, edgeList=self.edgeList, edgeTypeList=self.edgeTypeList,
                      afterCode=self.afterCode, fitnessDict=self.fitnessDict, reward=self.reward)
        self.actionTable: List[Action] = self.changeActionTableAccordingToAfterCode(
            needChangeActionTable=self.needChangeActionTable)
        state.actionTable = self.actionTable

        return state

    def step_macAss(self, actionIdx: int, preState_fitnessDict=None, needDisplay=0) -> State:
        # TODO step_macAss
        """

        Args:
            actionIdx: s_k___
            preState_fitnessDict: s_k_fitness
            needDisplay:

        Returns:

        """

        # 0.____local Search
        # 1.__cur_dgantt_code,A_B_jobIdCode__idx
        jobIdA: int = self.actionTable[actionIdx].jobIdA
        opIdA: int = self.actionTable[actionIdx].opIdA
        jobIdB: int = self.actionTable[actionIdx].jobIdB
        opIdB: int = self.actionTable[actionIdx].opIdB
        afterCode = self.afterCode
        jobCodeLen = len(afterCode) // 2
        # (jobId,macId)_____,___afterCodeIdx
        afterCodeIdx1 = self.end_dgantt.end_gantt.jobList[jobIdA][opIdA].afterCodeIdx
        afterCodeIdx2 = self.end_dgantt.end_gantt.jobList[jobIdB][opIdB].afterCodeIdx

        # 2.____lsCodeList

        # 2.1.____
        lsAfterCodeList = [copy.copy(self.afterCode) for _ in range(2)]
        lsAfterCodeList[1][afterCodeIdx1] = jobIdB
        lsAfterCodeList[1][afterCodeIdx2] = jobIdA

        lsAfterCodeList[0][afterCodeIdx1 + jobCodeLen] = -1
        lsAfterCodeList[0][afterCodeIdx2 + jobCodeLen] = -1
        lsAfterCodeList[1][afterCodeIdx1 + jobCodeLen] = -1
        lsAfterCodeList[1][afterCodeIdx2 + jobCodeLen] = -1

        # 2.____
        # dganttList
        dganttList = []
        # dganttList.append(self.end_dgantt)
        # _____(__,_____,__________jobIdOpId____)
        jobOpKey_macId__Dict = {}
        jobId_opIdInGanttFront = copy.copy(self.end_dgantt.jobId_opIdInGanttFront)
        for codeIdx in range(0, jobCodeLen):
            jobId = lsAfterCodeList[0][codeIdx]
            opId = jobId_opIdInGanttFront[jobId] + 1
            macId = lsAfterCodeList[0][codeIdx + jobCodeLen]
            jobId_opIdInGanttFront[jobId] += 1
            jobOpKey = jobId * self.maxConstant + opId
            jobOpKey_macId__Dict[jobOpKey] = macId
        # _______
        for idx, code in enumerate(lsAfterCodeList):
            # __code__
            macTime = [0 for _ in range(self.macN)]
            dgantt = self.end_dgantt.clone()
            for codeIdx in range(jobCodeLen):
                jobId = code[codeIdx]
                opId = dgantt.jobId_opIdInGanttFront_current[jobId] + 1
                jobOpKey = jobId * self.maxConstant + opId
                macId = jobOpKey_macId__Dict[jobOpKey]

                if macId == -1:
                    macIdTotalTimeDict = copy.copy(dgantt.end_gantt.jobList[jobId][opId].macIdTimeDict)
                    for macId, totalTime in enumerate(macTime):
                        if macId in macIdTotalTimeDict.keys():
                            macIdTotalTimeDict[macId] += totalTime
                    bestMacId, time = min(macIdTotalTimeDict.items(), key=lambda x: x[1])
                    code[codeIdx + jobCodeLen] = bestMacId
                    macId = bestMacId

                dgantt.appendNodeToEndGantt(jobId=jobId,
                                            macId=macId, afterCodeIdx=codeIdx)
                opId = dgantt.jobId_opIdInGanttFront_current[jobId]
                macTime[macId] += dgantt.end_gantt.jobList[jobId][opId].time

            dgantt.getDFitness()
            if needDisplay == 1:
                dgantt.displayDGantt()
                logprint.info(dgantt.end_gantt.fitness)
            dganttList.append(dgantt)

        # 3.__dganttList___
        # TODO ____reward
        bestDgantt = min(dganttList, key=lambda x: x.end_gantt.fitness)
        self.end_dgantt = bestDgantt
        # 4.__dgantt,_____
        self.setNodeEdgeState()
        self.setAfterCode()
        self.setFitnessDict()
        self.setReward(preState_fitnessDict)
        state = State(nodeList=self.nodeList, edgeList=self.edgeList, edgeTypeList=self.edgeTypeList,
                      afterCode=self.afterCode, fitnessDict=self.fitnessDict, reward=self.reward)
        self.actionTable: List[Action] = self.changeActionTableAccordingToAfterCode(
            needChangeActionTable=self.needChangeActionTable)
        state.actionTable = self.actionTable

        return state

    def step_jobSeq(self, actionIdx: int, preState_fitnessDict=None, needDisplay=0, returnBestResult=1) -> State:
        """
        local searching ____. ______,__job__,mac____
        Args:
            actionIdx: s_k___
            preState_fitnessDict: s_k_fitness
            needDisplay:
            returnBestResult:____job_seq________,____
        Returns:

        """
        # 0.____local Search
        # 1.__cur_dgantt_code,A_B_jobIdCode__idx
        jobIdA: int = self.actionTable[actionIdx].jobIdA
        opIdA: int = self.actionTable[actionIdx].opIdA
        jobIdB: int = self.actionTable[actionIdx].jobIdB
        opIdB: int = self.actionTable[actionIdx].opIdB
        afterCode = self.afterCode
        jobCodeLen = len(afterCode) // 2
        # (jobId,macId)_____,___afterCodeIdx
        afterCodeIdx1 = self.end_dgantt.end_gantt.jobList[jobIdA][opIdA].afterCodeIdx
        afterCodeIdx2 = self.end_dgantt.end_gantt.jobList[jobIdB][opIdB].afterCodeIdx

        # 2. ____
        # 2.1 __________(__,_____,__________jobIdOpId____)
        jobOpKey_macId__Dict = {}
        jobId_opIdInGanttFront = copy.copy(self.end_dgantt.jobId_opIdInGanttFront)
        for codeIdx in range(0, jobCodeLen):
            jobId = self.afterCode[codeIdx]
            opId = jobId_opIdInGanttFront[jobId] + 1
            macId = self.afterCode[codeIdx + jobCodeLen]
            jobId_opIdInGanttFront[jobId] += 1
            jobOpKey = jobId * self.maxConstant + opId
            jobOpKey_macId__Dict[jobOpKey] = macId
        # 2.2 ______
        # lsAfterCodeList = [copy.copy(self.afterCode) for _ in range(2)]
        lsAfterCode = copy.copy(self.afterCode)
        lsAfterCode[afterCodeIdx1] = jobIdB
        lsAfterCode[afterCodeIdx2] = jobIdA

        # 2.3 _________mac________
        # ________,_______
        macTime = [0 for _ in range(self.macN)]
        # clone()____________,____gantt
        dgantt = self.end_dgantt.clone()
        for codeIdx in range(jobCodeLen):
            jobId = lsAfterCode[codeIdx]
            opId = dgantt.jobId_opIdInGanttFront_current[jobId] + 1
            jobOpKey = jobId * self.maxConstant + opId
            macId = jobOpKey_macId__Dict[jobOpKey]
            dgantt.appendNodeToEndGantt(jobId=jobId,
                                        macId=macId, afterCodeIdx=codeIdx)
            opId = dgantt.jobId_opIdInGanttFront_current[jobId]
            macTime[macId] += dgantt.end_gantt.jobList[jobId][opId].time

        cur_fitness = dgantt.getDFitness()
        if needDisplay == 1:
            dgantt.displayDGantt()
            logprint.info(dgantt.end_gantt.fitness)

        # 3.__dgantt,______,_____
        if returnBestResult == 1:
            old_fitness = self.end_dgantt.getDFitness()
            if cur_fitness > old_fitness:
                return self.curState

        self.end_dgantt = dgantt
        self.setNodeEdgeState()
        self.setAfterCode()
        self.setFitnessDict()
        self.setReward(preState_fitnessDict)
        state = State(nodeList=self.nodeList, edgeList=self.edgeList, edgeTypeList=self.edgeTypeList,
                      afterCode=self.afterCode, fitnessDict=self.fitnessDict, reward=self.reward)
        self.actionTable: List[Action] = self.changeActionTableAccordingToAfterCode(
            needChangeActionTable=self.needChangeActionTable)
        state.actionTable = self.actionTable

        return state

    # 2.
    def setNodeEdgeState(self):
        "node.jobId, node.opId, node.macId, node.execId, node.time,+______"
        self.nodeList: List = []
        # __________:layer(x, torch.tensor([[],[]],dtype=torch.int), torch.tensor([]))
        self.edgeList: List = [[], []]
        self.edgeTypeList: List = []
        # __after_gantt__，__nodeList
        # 2.______,___
        nodeCount = 0
        for row in self.end_dgantt.end_gantt.jobList:
            for node in row:
                jobOpKey = node.jobId * self.maxConstant + node.opId
                if jobOpKey in self.end_dgantt.jobOpKey_remainNode.keys():
                    node.nodeId = nodeCount
                    nodeCount += 1

        for row in self.end_dgantt.end_gantt.jobList:
            for idx in range(1, len(row)):
                nodeA = row[idx - 1]
                nodeB = row[idx]
                jobOpKeyA = nodeA.jobId * self.maxConstant + nodeA.opId
                jobOpKeyB = nodeB.jobId * self.maxConstant + nodeB.opId
                if jobOpKeyA in self.end_dgantt.jobOpKey_remainNode.keys() and jobOpKeyB in self.end_dgantt.jobOpKey_remainNode.keys():
                    # self.edgeList.append([row[idx - 1].nodeId, row[idx].nodeId])
                    self.edgeList[0].append(row[idx - 1].nodeId)
                    self.edgeList[1].append(row[idx].nodeId)
                    self.edgeTypeList.append([0, row[idx].jobId])

        for row in self.end_dgantt.end_gantt.macList:
            for idx in range(1, len(row)):
                nodeA = row[idx - 1]
                nodeB = row[idx]
                jobOpKeyA = nodeA.jobId * self.maxConstant + nodeA.opId
                jobOpKeyB = nodeB.jobId * self.maxConstant + nodeB.opId
                if jobOpKeyA in self.end_dgantt.jobOpKey_remainNode.keys() and jobOpKeyB in self.end_dgantt.jobOpKey_remainNode.keys():
                    # self.edgeList.append([row[idx - 1].nodeId, row[idx].nodeId])
                    self.edgeList[0].append(row[idx - 1].nodeId)
                    self.edgeList[1].append(row[idx].nodeId)
                    self.edgeTypeList.append([1, row[idx].macId])

        # 4.__node__List
        for row in self.end_dgantt.end_gantt.macList:
            for idx, node in enumerate(row):
                jobOpKey = node.jobId * self.maxConstant + node.opId
                if jobOpKey in self.end_dgantt.jobOpKey_remainNode.keys():
                    # ___________
                    node.execId = idx
        for row in self.end_dgantt.end_gantt.jobList:
            for node in row:
                # _________
                jobOpKey = node.jobId * self.maxConstant + node.opId
                if jobOpKey in self.end_dgantt.jobOpKey_remainNode.keys():
                    # jobOpKey = node.jobId * self.maxConstant + node.opId
                    # jobOpListIdx = self.end_dgantt.start_gantt.instance.jobOpKey_jobOpListIdx[jobOpKey]
                    jobOpRow = self.end_dgantt.start_gantt.instance.jobOpList[node.jobId][node.opId]
                    # node.time_codeIdx___
                    jom = [node.jobId, node.opId, node.macId, node.execId, node.time] + list(
                        jobOpRow[0:len(jobOpRow)])
                    try:
                        self.nodeList.append(jom)
                    except:
                        logprint.info()
                        raise AttributeError

        return self.nodeList, self.edgeList, self.edgeTypeList

    def setFitnessDict(self):
        fitnessDict: Dict[str, float] = dict()
        makespan = self.end_dgantt.getDFitness()
        fitnessDict["makespan"] = makespan

        # TODO workload__
        workload = random.randint(0, 100)
        fitnessDict["workload"] = workload

        # print(makespan)
        self.fitnessDict = fitnessDict

    def setReward(self, preState_fitnessDict: Dict[str, float] = None):
        # fitness____, reward____
        self.reward = 0.0000000001
        if preState_fitnessDict is not None:
            self.reward = preState_fitnessDict["makespan"] - self.fitnessDict["makespan"]
        # for fitnessKey, fitnessVal in self.fitnessDict.items():
        #     if fitnessKey in self.rewardConfig:
        #         self.reward += fitnessVal * self.rewardConfig[fitnessKey]
        # self.reward = 1 / self.reward

    def setAfterCode(self):
        self.afterCode = self.end_dgantt.getAfterCode()
        return self.afterCode

    def changeActionTableAccordingToAfterCode(self, needChangeActionTable=1) -> List[Action]:
        # _____
        if needChangeActionTable == 0:
            return self.initialActionTable
        changeedActionTable: List[Action] = []
        for action in self.initialActionTable:
            keyA = action.jobIdA * self.maxConstant + action.opIdA
            keyB = action.jobIdB * self.maxConstant + action.opIdB
            nodeA = self.end_dgantt.jobOpKey_remainNode[keyA]
            nodeB = self.end_dgantt.jobOpKey_remainNode[keyB]
            if nodeA.macId != nodeB.macId:
                # ________
                continue
            if (nodeA.execId - nodeB.execId) ** 2 > 1:
                # ___
                continue
            changeedActionTable.append(action)
        return changeedActionTable

    def __enter__(self):
        # ________
        memory_info = psutil.virtual_memory()
        logprint.info(f"__env,________: {memory_info.percent}%")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ______，______，______
        # memory_info = psutil.virtual_memory()
        # logprint.info(f"__env,________: {memory_info.percent}%")
        # _______________，______、______
        # ___with_______，exc_type, exc_val, exc_tb_______
        del self
        gc.collect()
        if exc_type:
            print(f"____: {exc_type.__name__}, __: {exc_val}")
            # ______________，____True___，_____
            return False  # _____，_____
