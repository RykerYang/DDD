from typing import Dict


class GanttNode:

    def __init__(self, jobId=-1, opId=-1, startTime=0, macId=-1, time=0, hasInGantt=0, inBeforeDGantt: int = 0,
                 afterCodeIdx: int = -1):
        # 1.class Gantt __
        self.jobId = jobId
        self.opId = opId
        self.startTime = startTime
        self.macId = macId
        self.time = time  # ____
        self.hasInGantt: int = hasInGantt
        # self.macIdTimeDict[MacId]=time,__________
        self.macIdTimeDict: Dict[int, int] = dict()

        # 2.class DGantt __
        self.macPreNode: GanttNode = None
        # ___before_dgantt____.1___,0_______start_gantt____,_______
        # _______after_code <--> end_gantt_____
        self.instanceNodeId: int = -1
        # ___,_macList____?____execId
        self.execId = -1
        self.nodeId = -1
        self.afterCodeIdx = -1
        self.inBeforeDGantt = 0

    def __str__(self):
        rs = "jobId={},opId={},startTime={},macId={},time={},hasInGantt={}".format(self.jobId,
                                                                                   self.opId,
                                                                                   self.startTime,
                                                                                   self.macId,
                                                                                   self.time,
                                                                                   self.hasInGantt)
        return rs
