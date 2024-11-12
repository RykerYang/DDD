import datetime
import os
import pickle
from typing import Dict, Union, Tuple, List

import torch.nn
from torch.utils.tensorboard import SummaryWriter

from src.main.utils.Logger import logprint
from src.main.utils.Path import getResourcePath


# _____
class AlgoTimer:
    def __init__(self, resource_path):
        self.pickle_resource_path = os.path.join(resource_path, "pickle")
        if not os.path.exists(self.pickle_resource_path):
            os.mkdir(self.pickle_resource_path)
        # fileName,____
        self.objectDict: Dict[str, List[object]] = {}
        self.startTime = datetime.datetime.now()
        self.timePoints = []

    def setTimePoints(self, timePoints: List[float]):
        self.timePoints = timePoints

    def rightTimePoint(self):
        # 1.___, 2._______
        pass

    def setStartTime(self):
        self.startTime = datetime.datetime.now()

    # 2.level
    def getDuration(self) -> float:

        return (datetime.datetime.now() - self.startTime).total_seconds()

    def write(self, objectName: str, obj: object, needReset=0):
        obj_path = os.path.join(self.pickle_resource_path, objectName + ".pickle")
        if needReset == 1:
            self.objectDict[objectName] = [obj]
        else:
            if objectName not in self.objectDict.keys():
                # _____
                if os.path.exists(obj_path):
                    with open(obj_path, 'rb') as f:
                        self.objectDict[objectName] = pickle.load(f)
                    self.objectDict[objectName].append(obj)

                else:
                    self.objectDict[objectName] = [obj]

            else:

                self.objectDict[objectName].append(obj)

        with open(obj_path, 'wb') as f:
            pickle.dump(self.objectDict[objectName], f, pickle.HIGHEST_PROTOCOL)

    def read(self, objectName: str) -> List[object]:
        obj_path = os.path.join(self.pickle_resource_path, objectName + ".pickle")

        if objectName not in self.objectDict.keys():
            # _____
            if os.path.exists(obj_path):
                with open(obj_path, 'rb') as f:
                    self.objectDict[objectName] = pickle.load(f)


            else:
                return []

        return self.objectDict[objectName]

    def delete(self, objectName: str):
        obj_path = os.path.join(self.pickle_resource_path, objectName + ".pickle")
        if os.path.exists(obj_path):
            os.remove(obj_path)
        if objectName in self.objectDict.keys():
            del self.objectDict[objectName]


algoTimer = AlgoTimer(resource_path=getResourcePath())
