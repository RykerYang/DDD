import os
from typing import List
import torch.nn as nn
import torch
from torch.nn import Module

from src.main.servlet_rl.g_agent.model.DynamicNet import DynamicNet
from src.main.servlet_rl.g_agent.model.PredictNet import PredictNet
# from src.main2.servlet.ConfigManager import ConfigManager
from src.main.servlet_rl.g_agent.model.RepresentNet import RepresentNet
from src.main.servlet_rl.i_rlto.Action import Action
from src.main.utils.Logger import logprint
from src.main.utils.Path import getShareResourcePath


class Model(Module):
    def __init__(self, pqMode: int, representNet: RepresentNet, predictNet: PredictNet,
                 dynamicNet: DynamicNet, stepN: int = 0):
        """
        ______。

        Args:
            pqMode (int): P/Q____。
            stepN (int): RPD________。
            representNet (RepresentNet): ____。
            predictNet (PredictNet): ____。
            dynamicNet (DynamicNet): ____。
        """
        super().__init__()
        self.pqMode = pqMode
        self.stepN = stepN
        self.representNet = representNet
        self.predictNet: PredictNet = predictNet
        self.dynamicNet = dynamicNet

    def forward(self, nodeTensor, edgeTensor, edgeTypeTensor, actionTable: List[Action]):
        """
        ____。

        Args:
            nodeTensor: ____。
            edgeTensor: ___。
            edgeTypeTensor: _____。
            actionTensor: ____。

        Returns:
            action: ______。
        """
        state,available_mac_time_encodings = self.representNet(nodeTensor, edgeTensor, edgeTypeTensor)

        # for i in range(self.stepN):
        #     #_______
        #     next_state = self.dynamicNet(state, actionObjList[i])
        #     state = next_state
        action = self.predictNet(state, actionTable=actionTable, available_mac_time_encodings=available_mac_time_encodings)

        return action


class ModelManager:
    def __init__(self, representNetConfig, predictNetConfig, dynamicNetConfig, deviceName: str):
        """
        ________。

        Args:
            representNetConfig: ______。
            predictNetConfig: ______。
            dynamicNetConfig: ______。
        """
        self.deviceName: str = deviceName
        # self.actionTable: List[Action] = actionTable
        self.representNet: RepresentNet = RepresentNet(**representNetConfig).to(deviceName)
        self.predictNet: PredictNet = PredictNet(**predictNetConfig, deviceName=deviceName).to(deviceName)
        self.dynamicNet: DynamicNet = DynamicNet(**dynamicNetConfig).to(deviceName)
        self.model: Module
        # self.loader(getShareResourcePath(), 0)
        # self.loader(getShareResourcePath(), 1)
        # self.loader(getShareResourcePath(), 2)

    pass

    def saver(self, save_path: str,  save_net_type: int,save_file_name: str=""):
        """
        ____。

        Args:
            save_path: ____。
            save_file_name: _____。
            save_net_type (int): _______（0: ___，1: ___，2: ___）。
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # __ save_net_type __________
        if save_net_type == 0:
            model_to_save = self.representNet  # __ self.representation_model ____
        elif save_net_type == 1:
            model_to_save = self.predictNet  # __ self.prediction_model ____
        elif save_net_type == 2:
            model_to_save = self.dynamicNet  # __ self.dynamic_model ____
        elif save_net_type == 3:
            model_to_save = nn.ModuleList([self.predictNet.phead, self.predictNet.qhead])
        else:
            raise ValueError(f"Invalid save_net_type: {save_net_type}")

        if save_file_name=="":
            if save_net_type == 0:
                save_file_name = "representNet"  # __ self.representation_model ____
            elif save_net_type == 1:
                save_file_name = "predictNet"  # __ self.prediction_model ____
            elif save_net_type == 2:
                save_file_name = "dynamicNet"  # __ self.dynamic_model ____
            elif save_net_type == 3:
                save_file_name = "pq_head"
        # ________
        full_save_path = os.path.join(save_path, save_file_name+".pth")

        # ____
        torch.save(model_to_save.state_dict(), full_save_path)
        logprint.info(f"Model saved successfully at {full_save_path}")

    def loader(self, load_path: str,  load_net_type: int,load_file_name: str=""):
        """
        ____。

        Args:
            load_path: ____。
            load_file_name: _____。
            load_net_type (int): _______（0: ___，1: ___，2: ___）。
        """
        if not os.path.exists(load_path):
            os.makedirs(load_path)
        # __ load_net_type __________
        if load_net_type == 0:
            model_to_load = self.representNet  # __ self.representation_model ____
        elif load_net_type == 1:
            model_to_load = self.predictNet  # __ self.prediction_model ____
        elif load_net_type == 2:
            model_to_load = self.dynamicNet  # __ self.dynamic_model ____
        elif load_net_type == 3:
            model_to_load = nn.ModuleList([self.predictNet.phead,self.predictNet.qhead])
        else:
            raise ValueError(f"Invalid load_net_type: {load_net_type}")
        if load_file_name=="":
            if load_net_type == 0:
                load_file_name = "representNet"  # __ self.representation_model ____
            elif load_net_type == 1:
                load_file_name = "predictNet"  # __ self.prediction_model ____
            elif load_net_type == 2:
                load_file_name = "dynamicNet"  # __ self.dynamic_model ____
            elif load_net_type == 3:
                load_file_name = "pq_head"
        # ________
        full_load_path = os.path.join(load_path, load_file_name+".pth")

        # ____
        try:
            model_to_load.load_state_dict(torch.load(full_load_path))
            if load_net_type == 3:
                self.predictNet.phead=model_to_load[0]
                self.predictNet.qhead=model_to_load[1]
        except Exception as e:
            logprint.warning(f"Error loading model from {full_load_path}: {e}")
            return
        logprint.info(f"Model loaded successfully from {full_load_path}")

    def getModel(self, pqMode: int = 1, stepN: int = 0) -> Module:
        """
        ______。

        Args:
            pqMode (int): P/Q____，___1。{1:p__,2:q__,3:pq__}
            stepN (int): RPD________，___0。

        Returns:
            Model: ____。
        """

        model = Model(pqMode=pqMode, stepN=stepN, representNet=self.representNet, predictNet=self.predictNet,
                      dynamicNet=self.dynamicNet).to(self.deviceName)
        return model


if __name__ == '__main__':
    # todo ____,____
    pass
