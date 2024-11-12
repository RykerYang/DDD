import copy
import os
import pickle
import random
from typing import List, Dict

import numpy as np

from src.main.servlet_rl.h_env.e_dfjsp.f_DataBase import DataBase as DataBaseEnv
from src.main.servlet_rl.h_env.f_dgantt.f_Gantt import Gantt
from src.main.servlet_rl.h_env.f_dgantt.g_GanttNode import GanttNode
from src.main.servlet_rl.h_env.f_dgantt.g_Instance import Instance
from src.main.utils.Logger import logprint
from src.main.utils.Path import getResourcePath, getShareResourcePath


class DGantt:

    # 1.
    def __init__(self, start_gantt: Gantt, brokenMacId: int = -1,
                 stopTimePercent: float = 0, macRecoveryTime: int = 0,
                 maxConstant: int = 1000, random_time_seed_dynamic_event=-1):
        '''
        _____,_________
        :param gantt: _____,_______
        :param brokenMacId:
        :param stopTime:
        '''
        # 1.__________(start_gantt____)
        # ____
        self.maxConstant: int = maxConstant
        self.start_gantt: Gantt = start_gantt
        # 2.______
        self.brokenMacId: int = brokenMacId
        self.stopTime: float = self.start_gantt.fitness * stopTimePercent / 100
        self.stopTimePercent: float = stopTimePercent
        self.macRecoveryTime: int = macRecoveryTime

        # 3._________(before_gantt____)
        # 3.1______
        # jobId*self.maxConstant+opId:GanttNode
        self.jobOpKey_remainNode: Dict[int, GanttNode]
        # 3.1.1
        ## self.jobId_opIdInGanttFront_gantt__job____,
        ## ___op, self.jobId_opIdInGanttFront[jobId]=opId
        self.jobId_opIdInGanttFront: List[int]
        # self.end_gantt:_before_gantt--(____)-->>end_gantt
        self.before_gantt: Gantt = self.__setMacListDueToDynamicEvent__(
            self.stopTime)
        # 3.1.2 ______,_________,_______
        if brokenMacId != -1:
            self.__setRemainNodeCurrentBrokenMacId__(brokenMacId=brokenMacId)
        # 3.1.3
        if random_time_seed_dynamic_event != -1:
            self.__setRemainNodeCurrentRandomTime__(random_seed=random_time_seed_dynamic_event)
        # 3.2_____,____

        self.remainNodeN: int
        self.jobIdCode: List[int]
        self.jobId_jobIdCodeCount: List[int]
        self.__setBeforeGanttInfo__()

        # 4.___________(end_gantt____)
        temp = self.before_gantt.clone(self.jobOpKey_remainNode)
        self.end_gantt: Gantt = temp[0]
        self.jobOpKey_remainNode_current: Dict[int:GanttNode] = temp[1]
        self.jobId_opIdInGanttFront_current: List[int] = \
            copy.deepcopy(self.jobId_opIdInGanttFront)

        """
        self.jobOpKey_availableMacIdList:Dict[int:List[int]],
        _____GanttNode_macIdTimeDict__
        """

    def displayDGantt(self):
        self.end_gantt.displayGantt(
            "brokenMacId={},stopTime={},makespan={}".format(self.brokenMacId, self.stopTime,
                                                            self.end_gantt.fitness),
            macRecoveryTime=self.macRecoveryTime,
            stopTime=self.stopTime,
            brokenMac=self.brokenMacId)

    def getDFitness(self):
        self.end_gantt.adjustGantt()
        self.end_gantt.__setFitness__()
        return self.end_gantt.fitness

    # def getEndCode(self):
    #     assert len(self.jobOpKey_remainNode_current) == 0
    #     return self.end_gantt.getEndCodeByGanttUsingTopo()

    def getAfterCode(self, needSetGanttNodeAfterCodeIdx=1) -> List[int]:
        # 0.__afterCode
        assert len(self.jobOpKey_remainNode_current) == 0
        # _______self.getDFitness()
        # try:
        #     assert self.end_gantt.jobList[0][-1].startTime != 0
        # except:
        #     logprint.error(f"{str(self.end_gantt.jobList)}")
        #     raise AssertionError("")

        ganttNodeList = []

        for jobOpKey in self.jobOpKey_remainNode.keys():
            jobId = jobOpKey // self.maxConstant
            opId = jobOpKey % self.maxConstant
            ganttnode = self.end_gantt.jobList[jobId][opId]
            assert ganttnode.afterCodeIdx != -1
            ganttNodeList.append(ganttnode)

        ganttNodeList = sorted(ganttNodeList, key=lambda x: x.afterCodeIdx)

        endJobCode = []
        endMacCode = []
        for afterCodeIdx, node in enumerate(ganttNodeList):
            endJobCode.append(node.jobId)
            endMacCode.append(node.macId)
            if needSetGanttNodeAfterCodeIdx == 1:
                node.afterCodeIdx = afterCodeIdx

        return endJobCode + endMacCode

    def clone(self, needReset=1):
        #
        dgantt = copy.copy(self)
        # dgantt.end_gantt = self.before_gantt.clone()
        if needReset == 1:
            dgantt.jobId_opIdInGanttFront_current = \
                copy.deepcopy(self.jobId_opIdInGanttFront)
            # dgantt.__setEndGanttInfo__()
            temp = self.before_gantt.clone(self.jobOpKey_remainNode)
            dgantt.end_gantt = temp[0]
            dgantt.jobOpKey_remainNode_current = temp[1]
        else:
            dgantt.jobId_opIdInGanttFront_current = \
                copy.deepcopy(self.jobId_opIdInGanttFront_current)
            # dgantt.__setEndGanttInfo__()
            temp = self.end_gantt.clone(self.jobOpKey_remainNode_current)
            dgantt.end_gantt = temp[0]
            dgantt.jobOpKey_remainNode_current = temp[1]

        return dgantt

    def getCurRemainedNodeN(self) -> int:
        return len(self.jobOpKey_remainNode_current.keys())

    def appendNodeToEndGantt(self, jobId: int, opId=-1, macId=-1,
                             needAdjustGanttSetFitness=0, afterCodeIdx=-1):

        # 1.___job,op,mac____
        if opId == -1:
            try:
                opId = self.jobId_opIdInGanttFront_current[jobId] + 1
            except:
                raise TypeError()
        jobOpKey = jobId * self.maxConstant + opId
        try:
            assert jobOpKey in self.jobOpKey_remainNode_current.keys()
        except:
            raise ValueError()

        if macId == -1:
            macId = self.jobOpKey_remainNode_current[jobOpKey].macId
        try:
            assert macId in self.jobOpKey_remainNode_current[jobOpKey].macIdTimeDict.keys()
        except:
            raise ValueError()
        # 2._jobList_remainNode(______),__node
        node: GanttNode = self.jobOpKey_remainNode_current[jobOpKey]
        # 2.1_______(___),____
        del self.jobOpKey_remainNode_current[jobOpKey]

        # 3.______
        node.macId = macId
        node.time = node.macIdTimeDict[macId]
        node.hasInGantt = 1
        node.afterCodeIdx = afterCodeIdx

        if len(self.end_gantt.macList[node.macId]) >= 2:
            node.macPreNode = self.end_gantt.macList[node.macId][-2]
        else:
            node.macPreNode = None
        # 4.__startTime
        if node.macId == self.brokenMacId:
            node.startTime = self.stopTime + self.macRecoveryTime
        else:
            node.startTime = self.stopTime

        # 5.____macList

        self.end_gantt.macList[node.macId].append(node)
        self.jobId_opIdInGanttFront_current[jobId] += 1
        # 6.__MacList
        if needAdjustGanttSetFitness == 1:
            self.end_gantt.adjustGantt()
            self.end_gantt.__setFitness__()

    def fixAfterCode(self, afterCode, maxConstant=1000):
        # 1 __________(__,_____,__________jobIdOpId____)
        jobCodeLen = len(afterCode) // 2
        assert len(self.jobOpKey_remainNode_current.keys()) <= jobCodeLen, "afterCode_______"
        jobOpKey_macId__Dict = {}
        jobId_opIdInCurAfterCode = [len(row) - 1 for row in self.start_gantt.instance.jobOpList]
        for codeIdx in reversed(range(0, jobCodeLen)):
            jobId = afterCode[codeIdx]
            opId = jobId_opIdInCurAfterCode[jobId]
            macId = afterCode[codeIdx + jobCodeLen]
            jobId_opIdInCurAfterCode[jobId] -= 1
            try:
                assert jobId_opIdInCurAfterCode[jobId] >= -1, "jobId_opIdInCurAfterCode[jobId]<0"
            except AssertionError:
                logprint.error(f"{jobId_opIdInCurAfterCode}")
                raise AssertionError()
            jobOpKey = jobId * self.maxConstant + opId
            jobOpKey_macId__Dict[jobOpKey] = macId
        # 2.____________remainNode,__afterCode
        # __remainNode
        right_jobOpKey_macId__Dict = {}
        for jobOpKey in self.jobOpKey_remainNode_current.keys():
            if jobOpKey in jobOpKey_macId__Dict.keys():
                macId = jobOpKey_macId__Dict[jobOpKey]
                if macId not in self.jobOpKey_remainNode_current[jobOpKey].macIdTimeDict.keys():
                    # todo logprint.warning("___________")

                    # _______
                    macId = min(self.jobOpKey_remainNode_current[jobOpKey].macIdTimeDict.items(), key=lambda x: x[1])[0]

                    # macId = list(self.jobOpKey_remainNode_current[jobOpKey].macIdTimeDict.keys())[0]
                right_jobOpKey_macId__Dict[jobOpKey] = macId
            else:
                raise ValueError("___aftercode__")

        # 3.
        jobCode = []
        macCode = []
        for key, macId in sorted(right_jobOpKey_macId__Dict.items(), key=lambda x: x[0]):
            jobCode.append(key // maxConstant)
            macCode.append(macId)

        return jobCode + macCode

    # 2.
    def __setMacListDueToDynamicEvent__(self, stopTime) -> Gantt:
        # ____
        if not isinstance(self.start_gantt.macList, list):
            raise TypeError()
        if len(self.start_gantt.macList) == 0:
            raise ValueError()
        # if len(self.start_gantt.macList[0]) == 0:
        #     raise ValueError()
        #
        # if not isinstance(self.start_gantt.macList[0], list):
        #     raise TypeError()
        # if not isinstance(self.start_gantt.macList[0][0], GanttNode):
        #     raise TypeError()
        # __
        end_gantt = self.start_gantt.clone()
        macList = end_gantt.macList
        jobList = end_gantt.jobList

        # __
        # 1. __stopTime__,______.

        ## 1.1 ____,__hasInGantt = 0
        for row in macList:
            for node in row:
                if node.startTime >= stopTime:
                    node.hasInGantt = 0

                # elif node.macId == brokenMacId:
                #     endTime = node.startTime + node.time
                #     if endTime > stopTime:
                #         # ___,_____,
                #         # _____stopTime,________
                #         node.hasInGantt = 0
        ## 1.2 jobOpKey_remainNode_______
        jobOpKey_remainNode: Dict[int:GanttNode] = dict()
        for row in macList:
            for node in row:
                if node.hasInGantt == 0:
                    key = self.maxConstant * node.jobId + node.opId
                    node.startTime = self.stopTime
                    jobOpKey_remainNode[key] = node

        ## 1.3 macList_________
        for row in macList:
            idx = 0
            while idx < len(row):
                node = row[idx]
                if node.hasInGantt == 0:
                    del row[idx]
                else:
                    idx += 1

        ## 1.4 __node.inBeforeGantt
        for row in jobList:
            for node in row:
                node.inBeforeDGantt = 0
        for row in macList:
            for node in row:
                node.inBeforeDGantt = 1

        # 2. __jobId_opIdInGanttFront
        jobId_opIdInGanttFront = [-1 for _ in range(end_gantt.instance.jobN)]
        for row in macList:
            for node in row:
                jobId = node.jobId
                opId = node.opId
                if jobId_opIdInGanttFront[jobId] < opId:
                    jobId_opIdInGanttFront[jobId] = opId

        # __
        end_gantt.macList = macList
        self.jobId_opIdInGanttFront = jobId_opIdInGanttFront
        self.jobOpKey_remainNode = jobOpKey_remainNode

        return end_gantt

    def __setBeforeGanttInfo__(self):
        self.remainNodeN: int = len(self.jobOpKey_remainNode.keys())
        self.jobIdCode: List[int] = [node.jobId for node in self.jobOpKey_remainNode.values()]
        self.jobId_jobIdCodeCount: List[int] = [0 for _ in range(self.start_gantt.instance.jobN)]
        for jobId in self.jobIdCode:
            self.jobId_jobIdCodeCount[jobId] += 1

    def __setRemainNodeCurrentBrokenMacId__(self, brokenMacId: int = -1):
        """
        ______,__self.jobOpKey_remainNode
        Args:
            brokenMacId:-1__,______broken.
                ___Env__,env____________,________!!

        Returns:

        """
        if brokenMacId < 0:
            return
        needDeleteKeys = []
        for key, node in self.jobOpKey_remainNode.items():
            if brokenMacId in node.macIdTimeDict.keys():
                # try:
                del node.macIdTimeDict[brokenMacId]
                if len(node.macIdTimeDict) == 0:
                    needDeleteKeys.append(key)
        # __________,__________
        count = len(self.jobOpKey_remainNode)
        jobOpKey_remainNode_keys = list(self.jobOpKey_remainNode.keys())
        while count > 0:
            jobOpKey_remainNode_key = jobOpKey_remainNode_keys[-count]
            for key in needDeleteKeys:
                if key == jobOpKey_remainNode_key:
                    try:
                        del self.jobOpKey_remainNode[jobOpKey_remainNode_key]
                    except Exception as ex:
                        pass
                    continue
                if jobOpKey_remainNode_key > key and key // self.maxConstant == (
                        jobOpKey_remainNode_key // self.maxConstant):
                    try:
                        del self.jobOpKey_remainNode[jobOpKey_remainNode_key]
                    except Exception as ex:
                        logprint.info(ex)
                    continue

            count -= 1

    def __setRemainNodeCurrentRandomTime__(self, random_seed=-1):
        if random_seed >= 0:
            random.seed(random_seed)
        for key, node in self.jobOpKey_remainNode.items():
            for macId, time in node.macIdTimeDict.items():
                random_time_rate = random.uniform(0, 1)
                needIncreasingTime = random.randint(0, 1)
                if needIncreasingTime:
                    node.macIdTimeDict[macId] = time + int(time * random_time_rate)
                else:
                    node.macIdTimeDict[macId] = time - int(time * random_time_rate)
        # ______
        random.seed()


def randomDynamicEvent(instanceId, start_codeId, size=5, needSave=0):
    dynamicEvents_startCodeId1_fileName = os.path.join(getResourcePath(), 'dynamicEvents.pickle')
    dynamicEvents_startCodeId1: List[List[int or float]] = list()
    # 1.__start_gantt
    dataBaseRL = DataBaseEnv(instanceId=instanceId, shareResourcePath=getShareResourcePath())
    jobOpList, start_code = dataBaseRL.query_instance(start_codeId=start_codeId)
    instance: Instance = Instance(jobOpList=jobOpList)
    start_gantt: Gantt = Gantt(start_code=start_code,
                               instance=instance)
    # 2._mac_brokenProbality
    mac_brokenProbality: List[float] = [0 for macId in range(instance.macN)]
    MBT_total = 0
    MBT_k: List[int] = [0 for macId in range(instance.macN)]
    for row in start_gantt.macList:
        for node in row:
            MBT_k[node.macId] += node.time
            MBT_total += node.time

    for macId in range(instance.macN):
        mac_brokenProbality[macId] = MBT_k[macId] / MBT_total
    # 3.__brokenMacId
    brokenMacIds = sorted(
        list(np.random.choice(list(range(instance.macN)), size=size, p=mac_brokenProbality)))
    stopTimePercents = list(np.random.uniform(0, 100, size=size))
    for i in range(len(brokenMacIds)):
        dynamicEvents_startCodeId1.append(
            # __id, _______, __________fitness
            [brokenMacIds[i], stopTimePercents[i] * 100 // 1 / 100, stopTimePercents[i] / 100 * start_gantt.fitness* 100 // 1 / 100])
    # print(dynamicEvents_startCodeId1)
    if needSave == 1:
        with open(dynamicEvents_startCodeId1_fileName, 'wb') as f:
            pickle.dump(dynamicEvents_startCodeId1, f)

    return dynamicEvents_startCodeId1
