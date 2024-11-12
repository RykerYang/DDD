import copy
from typing import List, Dict

import matplotlib.pyplot as plt

from src.main.servlet_rl.h_env.f_dgantt.g_GanttNode import GanttNode
from src.main.servlet_rl.h_env.f_dgantt.g_Instance import Instance
from src.main.utils.Logger import logprint


class Gantt:
    # 1.
    def __init__(self, start_code: List[int], instance: Instance):
        assert isinstance(start_code, list)
        # assert len(start_code) == instance.jobOpGN * 2

        if len(start_code) != instance.jobOpGN * 2:
            raise AttributeError

        self.macList: List[List[GanttNode]] = [[] for i in range(instance.macN)]
        # __, _ganttnode___
        self.jobList: List[List[GanttNode]] = [[] for i in range(instance.jobN)]
        self.fitness: int = -1

        self.instance = instance
        self.start_code = start_code

        # ___
        self.__initJobListByInstance__()
        self.__initMacListByStartCode__()
        self.__setFitness__()

    def resetCode(self, code):
        self.macList: List[List[GanttNode]] = [[] for i in range(self.instance.macN)]
        self.start_code = code
        self.__initMacListByStartCode__()
        self.__setFitness__()

        return self.getFitness(needUpdateFitness=0)

    def getEndCodeByGanttUsingTopo(self) -> List[int]:
        assert self.fitness != -1

        nodeList = []
        for row in self.jobList:
            for node in row:
                nodeList.append(node)

        sorted(nodeList, key=lambda x: x.startTime)

        endJobCode = []
        endMacCode = []
        for node in nodeList:
            endJobCode.append(node.jobId)
            endMacCode.append(node.macId)

        return endJobCode + endMacCode

    def getFitness(self, needUpdateFitness=1):
        if needUpdateFitness == 1:
            self.__setFitness__()
        return self.fitness

    def copy(self):
        # __instance_____
        instance = self.instance
        self.instance = None
        gantt_copy = copy.deepcopy(self)
        gantt_copy.instance = instance
        self.instance = instance
        return gantt_copy

    def displayGantt(self, title="Gantt", macRecoveryTime=-1, stopTime=-1, brokenMac=-1):
        # _________？
        # __self.macList___。_____,self.macList______,displayGantt()_____？__,____________,_______,self.macList___,________
        jobN = self.instance.jobN
        macN = self.instance.macN

        # import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        plt.rcParams['figure.dpi'] = 700
        colorList = ['#0000CD', '#008B8B', '#00CED1', '#00FF7F', '#104E8B',
                     '#1C86EE', 'tab:pink', '#7CCD7C', '#7FFFD4', '#8968CD',
                     '#3ec7a5', '#f00e68', '#ffff00', '#f9cb9c', '#00ffff',
                     '#ff00ff', '#cc6601', '#9900f0', '#3475D0', '#ffccff']
        # Horizontal bar plot with gaps
        fig, ax = plt.subplots()

        for row in self.macList:
            for elem in row:
                # y_mac
                if elem.jobId >= 10:
                    # broken_barh(xranges=[(x__,_)], yrange=(y__,_))
                    ax.broken_barh([(elem.startTime, elem.time)], (elem.macId, 0.7), facecolor=colorList[elem.jobId],
                                   edgecolor="black", linestyle="-.")
                else:

                    ax.broken_barh([(elem.startTime, elem.time)], (elem.macId, 0.7), facecolor=colorList[elem.jobId],
                                   edgecolor="black", linestyle="-")

                plt.text(x=elem.startTime, y=elem.macId + 0.75, s=str(elem.opId) + "/", fontsize=2)
                plt.text(x=elem.startTime, y=elem.macId, s=str(elem.jobId) + "/", fontsize=2)

        # __y___

        ax.set_ylim(0, macN)
        # __x___
        ax.set_xlim(0, self.fitness * 1.3)
        ax.set_xlabel('seconds since start')
        ax.set_yticks([i for i in range(0, macN, 1)], labels=[i for i in range(0, macN)])  # Modify y-axis tick labels
        ax.grid(True)  # Make grid lines visible
        legend_elements = [Patch(facecolor=elem, label=idx) for idx, elem in enumerate(colorList)][0:jobN]
        plt.legend(handles=legend_elements, loc=1)

        plt.xlabel('makespan')
        if macRecoveryTime != -1:
            ax.broken_barh([(stopTime, macRecoveryTime)], (brokenMac, 0.7), facecolor="#FFFFFF",
                           edgecolor="black", linestyle="--")
            plt.text(x=stopTime, y=brokenMac, s="recovery", fontsize=5)

        if title != "":
            plt.title(title)

        plt.show()

    def clone(self, jobOpKey_remainNode: Dict = None):
        gantt = copy.copy(self)
        temp = [self.jobList, self.macList, self.start_code]
        if jobOpKey_remainNode != None:
            temp.append(jobOpKey_remainNode)
        temp = copy.deepcopy(temp)
        gantt.jobList = temp[0]
        gantt.macList = temp[1]
        gantt.start_code = temp[2]

        if jobOpKey_remainNode != None:
            return (gantt, temp[3])

        return (gantt)

    # 2.
    def __initJobListByInstance__(self):
        # __
        jobOpList = self.instance.jobOpList
        jobList = self.jobList

        # __

        # 1.__jobList,job__
        # __instance___jobList
        for jobId, jobRow in enumerate(jobOpList):
            for opId, opRow in enumerate(jobRow):
                node = GanttNode()
                node.jobId = jobId
                node.opId = opId
                # __macIdTimeDict
                avaliableMac = opRow
                for macId, time in enumerate(avaliableMac):
                    if time != -1:
                        node.macIdTimeDict[macId] = time

                jobList[node.jobId].append(node)

        # __
        self.jobList = jobList

    def __initMacListByStartCode__(self):
        # __
        jobOpN = self.instance.jobOpGN
        # 0.__:__gantt_
        # 1.__code__

        # *____jobId_opId
        jobOpId_count = [0 for _ in range(0, self.instance.jobN)]
        for codeId in range(0, len(self.start_code) // 2):
            # 1.1 _____(jobId,opId,macId)
            jobId = int(self.start_code[codeId])
            opId = jobOpId_count[jobId]
            jobOpId_count[jobId] += 1
            macId = self.start_code[jobOpN + codeId]
            # 1.2 __instance,_________
            if self.instance.jobOpList[jobId][opId][macId] == -1:
                logprint.info(f"{jobId}-{opId}-{macId}")
                raise ValueError("code__")

        # 2.___macList
        # *____jobId_opId
        jobOpId_count = [0 for _ in range(0, self.instance.jobN)]
        for codeId in range(0, len(self.start_code) // 2):
            # 2.1 _____(jobId,opId,macId)
            jobId = int(self.start_code[codeId])
            opId = jobOpId_count[jobId]
            jobOpId_count[jobId] += 1
            macId = self.start_code[jobOpN + codeId]
            # 2.2 __node
            node = self.jobList[jobId][opId]
            node.macId = macId
            node.hasInGantt = 1
            node.startTime = 0
            # 2.3 __node.time
            time = self.instance.jobOpList[jobId][opId][macId]
            node.time = time
            # 2.4 __macList
            self.macList[node.macId].append(node)

        # 3.__MacList
        self.adjustGantt()

        self.__setFitness__()
        # __
        # self.fitness =
        # self.macList=

    def __setFitness__(self):
        # __
        # self.macList[0][0]___
        macListNodeN = sum([len(row) for row in self.macList])
        if macListNodeN == 0:
            self.fitness = 0
            return

            # raise ValueError("macList____")
        makespan = 0
        for row in self.macList:
            # __row[-1]__
            if len(row) == 0:
                continue

            endTime = row[-1].startTime + row[-1].time
            if endTime > makespan:
                makespan = endTime

        # __
        self.fitness = makespan

    def adjustGantt(self):
        hasRightGanttList = 0
        hasRightJobList = 0
        while hasRightJobList == 0 or hasRightGanttList == 0:
            hasRightGanttList = self.__adjustNode__(self.macList)
            hasRightJobList = self.__adjustNode__(self.jobList)

    # 3.
    def __adjustNode__(self, jobOrMacList):
        # ____
        if isinstance(jobOrMacList, list) == False:
            raise TypeError

        hasRight = 1
        for instance in jobOrMacList:
            endTime = 0
            for jobOpNode in instance:
                if endTime <= jobOpNode.startTime:
                    pass
                else:
                    hasRight = 0
                    jobOpNode.startTime = endTime
                endTime = jobOpNode.startTime + jobOpNode.time
        return hasRight
