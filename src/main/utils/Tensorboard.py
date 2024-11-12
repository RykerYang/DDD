import datetime
import os
from typing import Dict, Union, Tuple

import torch.nn
from torch.utils.tensorboard import SummaryWriter

from src.main.utils.Logger import logprint
from src.main.utils.Path import getResourcePath


class Tensorboard:
    def __init__(self, log_dir):
        current_time = datetime.datetime.now()
        tag_time = current_time.strftime('%Y%m%d_%H%M%S')

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, tag_time))
        # tag_name:stepCount
        self.tagDict: Dict[str, int] = {}
        self.log_dir = log_dir

    def graph(self, model: torch.nn.Module, input_to_model: Union[Tuple[torch.Tensor], torch.Tensor]):
        self.writer.add_graph(model, input_to_model)

    def scalar(self, tag_name: str, value: Union[float, Dict[str, float]]):
        if tag_name not in self.tagDict:
            self.create_tag(tag_name)

        if isinstance(value, dict):
            self.writer.add_scalars(tag_name, value, self.tagDict[tag_name])
        elif isinstance(value, float):
            self.writer.add_scalar(tag_name, value, self.tagDict[tag_name])
        else:
            raise TypeError("value must be float or dict")
        # logprint.info(f"tag_name:{tag_name}, value:{value},step:{self.tagDict[tag_name]}")
        self.tagDict[tag_name] += 1

    # level 2
    def create_tag(self, tag_name: str):
        self.tagDict[tag_name] = 0

    def reset(self):
        self.writer.close()

        current_time = datetime.datetime.now()
        tag_time = current_time.strftime('%Y%m%d_%H%M%S')

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, tag_time))
        # tag_name:stepCount
        self.tagDict: Dict[str, int] = {}



# tensorboard --logdir=runs
boardprint = Tensorboard(os.path.join(getResourcePath(), "runs"))
