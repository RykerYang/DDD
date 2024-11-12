class Action:
    def __init__(self, actionIdx: int, nodeIdxA: int, jobIdA: int, opIdA: int, nodeIdxB: int, jobIdB: int, opIdB: int):
        self.actionIdx: int = int(actionIdx)
        self.nodeIdxA: int = int(nodeIdxA)
        self.jobIdA: int = int(jobIdA)
        self.opIdA: int = int(opIdA)
        self.nodeIdxB: int = int(nodeIdxB)
        self.jobIdB: int = int(jobIdB)
        self.opIdB: int = int(opIdB)
