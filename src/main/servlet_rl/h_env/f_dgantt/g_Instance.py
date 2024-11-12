import random
from typing import List

import numpy as np


class Instance:
    def __init__(self, jobOpList: List[List[List[int]]]):

        if not isinstance(jobOpList, list):
            raise TypeError()
        if len(jobOpList) == 0:
            raise ValueError()
        if not isinstance(jobOpList[0], list):
            raise TypeError()
        # if len(jobOpList[0]) <= 3:
        #     raise ValueError()
        for opRow in jobOpList:
            for row in opRow:
                for elem in row:
                    if not isinstance(elem, int):
                        raise TypeError()
        # ____jobOp___________
        for jobId, job in enumerate(jobOpList):
            for opId, op in enumerate(job):
                macTimeList = jobOpList[jobId][opId]
                availableMacIdList = list()
                for macId, time in enumerate(macTimeList):
                    if time != -1:
                        availableMacIdList.append(macId)
                if len(availableMacIdList) == 0:
                    raise ValueError(f"jobId-{jobId} opId-{opId} has no available macId")

        # 3.jobOpList[jobId][opId][macId:_0__]
        self.jobOpList = jobOpList
        # 4.
        self.jobN = len(self.jobOpList)
        # 5.
        self.macN = len(jobOpList[0][0])

        # 6.obOpList_op__
        self.jobOpGN = 0
        for jobId, opRow in enumerate(self.jobOpList):
            self.jobOpGN += len(opRow)
        # 7.
        self.jobCodeLen = self.jobOpGN
        # 8.
        self.macCodeLen = self.jobOpGN

        # 10.code__:jobCode,macCode,factoryCode
        self.codeLen = self.jobCodeLen + self.macCodeLen

        timeMax = -1
        for jobId, opRow in enumerate(self.jobOpList):
            for opId, macRow in enumerate(opRow):
                time = max(self.jobOpList[jobId][opId])
                if max(self.jobOpList[jobId][opId]) > timeMax:
                    timeMax = time
        # 11.
        self.timeMax = timeMax

    def getRandomCode(self, random_seed=-1) -> List[int]:
        """
        ________。

        ________，___________（mac）ID，___________ID____mac ID___。

        :__: code: list - ____ID____mac ID_____。
        """
        if random_seed != -1:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # 1. init jobSeq
        # ________mac____
        jobSeq = []
        for jobId, job in enumerate(self.jobOpList):
            for opId, op in enumerate(job):
                jobSeq.append(jobId)
        random.shuffle(jobSeq)
        # 2. init macAssign
        macAssign = []
        jobOpId_count = [0 for _ in range(0, self.jobN)]
        for codeIdx, jobId in enumerate(jobSeq):
            # 1.1 _____(jobId,opId,macId)
            opId = jobOpId_count[jobId]
            jobOpId_count[jobId] += 1
            # _______mac ID
            macTimeList = self.jobOpList[jobId][opId]
            availableMacIdList = list()
            for macId, time in enumerate(macTimeList):
                if time != -1:
                    availableMacIdList.append(macId)
            # _________mac ID
            if len(availableMacIdList) == 0:
                raise ValueError("_____macId")
            target_macId = random.choice(availableMacIdList)
            macAssign.append(target_macId)
            # logprint.info(f"{jobId}-{opId}-{target_macId}-{self.jobOpList[jobId][opId][target_macId]}-{codeIdx}-{len(macAssign)}")

        # 3._________
        code = jobSeq + macAssign
        # logprint.info(f"jobSeq:{jobSeq},len:{len(jobSeq)}")
        # logprint.info(f"macAssign:{macAssign},len:{len(macAssign)}")

        return code


def generateRamdomInstance(maxJobN, maxOpN, maxMacN, maxTime, minJobN=1, minOpN=2, minMacN=1, random_seed=-1):
    # min<=x<max
    # ________
    assert maxJobN > 1, "maxJobN must > 1"
    # _______2_op,______action!______!!
    assert maxOpN > 2, "maxOpN must > 2"
    assert maxMacN > 1, "maxMacN must > 1"
    assert maxTime > 0
    assert minJobN > 0, "minJobN must > 0"
    assert minOpN > 1, "minOpN must > 1"
    assert minMacN > 0, "minMacN must > 0"
    assert maxJobN > minJobN, "maxJobN must > minJobN"
    assert maxOpN > minOpN, "maxOpN must > minOpN"
    assert maxMacN > minMacN, "maxMacN must > minMacN"

    if random_seed != -1:
        random.seed(random_seed)
        np.random.seed(random_seed)
    # __Instance
    jobOpList = []
    macN = np.random.randint(minMacN, maxMacN)
    for jobId in range(np.random.randint(minJobN, maxJobN)):
        opList = []
        for opId in range(np.random.randint(minOpN, maxOpN)):
            timeList = [np.random.randint(1, maxTime) for macId in range(macN - 1)]
            for timeIdx, time in enumerate(timeList):
                isNegOne = np.random.randint(0, 1)
                if isNegOne == 1:
                    timeList[timeIdx] = -1
            # ____jobOp___________
            if macN - 1 > 0:
                timeList.insert(np.random.randint(0, macN - 1), np.random.randint(1, maxTime - 1))
            else:
                timeList.append(np.random.randint(1, maxTime - 1))

            opList.append(timeList)
        jobOpList.append(opList)

    return Instance(jobOpList)
