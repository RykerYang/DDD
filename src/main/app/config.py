from typing import List

from src.main.servlet_rl.i_rlto.Action import Action
from src.main.utils.Path import getShareResourcePath

import_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),

)


# deviceName = "cpu"
# state_dim = 64
# macN = -1
#
# actionTable:List[Action]=[]
# # _______
# jobOpGN = -1
#
# modelManagerConfig = dict(
#     representNetConfig=dict(
#         nodeEncoderConfig=dict(
#             hidden_channels=64,
#             nodeBin2_macId_len=macN,
#             nodeBin3_codeIdx_len=jobOpGN,
#             deviceName=deviceName,
#         ),
#
#         structureEncoderConfig=dict(
#             hidden_channels=state_dim,
#             num_layers=10,
#             edgeTypeN=2
#         )
#     ),
#     predictNetConfig=dict(
#         actionTable=actionTable,
#         # phead_qhead_____
#         deviceName=deviceName,
#         pheadConfig=dict(
#             # hidden_channels * 2
#             in_dim=state_dim * 2,
#             # default:1
#             out_dim=1,
#             # default:5
#             nLayer=5
#         ),
#         qheadConfig=dict(
#             # hidden_channels * 3
#             in_dim=state_dim * 2,
#             out_dim=1,
#             # default:5
#             nLayer=5
#         ),
#     ),
#     dynamicNetConfig=dict(
#         in_out_dim=state_dim,
#         # default:1
#         nLayer=1
#
#     )
#
# )
def getEnvConfig(stopTimePercent=0,
                 brokenMacId=-1,
                 macRecoveryTime=0,
                 instanceId=1,
                 start_codeId=3) -> dict:
    env = dict(
        stopTimePercent=stopTimePercent,
        brokenMacId=brokenMacId,
        macRecoveryTime=macRecoveryTime,
        instanceId=instanceId,
        start_codeId=start_codeId,
        maxConstant=1000,
        shareResourcePath=getShareResourcePath(),
        rewardConfig={"makespan": 1.0}
    )
    return env


def getModelManagerConfig(macN: int, jobOpGN: int, actionTable: List[Action], state_dim=50, deviceName="cpu") -> dict:
    # deviceName = deviceName
    state_dim = state_dim
    macN = macN
    # actionTable: List[Action] = actionTable

    # _______
    jobOpGN = jobOpGN
    modelManagerConfig = dict(
        deviceName=deviceName,
        # actionTable=actionTable,
        representNetConfig=dict(
            attr_dim=10,
            edge_attr_dim=6
            # nodeEncoderConfig=dict(
            #     # ______,_________
            #     hidden_channels=state_dim,
            #     nodeBin2_macId_len=macN,
            #     nodeBin3_codeIdx_len=jobOpGN,
            # ),
            #
            # structureEncoderConfig=dict(
            #     # _________________
            #     hidden_channels=state_dim,
            #     num_layers=10,
            #     edgeTypeN=2
            # )
        ),
        predictNetConfig=dict(
            # phead_qhead_____
            pheadConfig=dict(
                # hidden_channels * 2
                in_dim=state_dim+10 ,
                # default:1
                out_dim=1,
                # default:5
                nLayer=5
            ),
            qheadConfig=dict(
                # hidden_channels * 3
                in_dim=state_dim +10,
                out_dim=1,
                # default:5
                nLayer=5
            ),
        ),
        dynamicNetConfig=dict(
            in_out_dim=state_dim,
            # default:1
            nLayer=1

        )

    )

    return modelManagerConfig
