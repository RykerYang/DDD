from typing import List


class IPipeline:

    def __init__(self):
        self.rlContextList: List = None



    def collecting(self,playoutN=1):
        raise NotImplementedError()
    def learning(self,trainN=10, samplePercentN=0.5,deviceName="cpu"):
        # deviceName__cpu
        raise NotImplementedError()

    def evaluating(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

