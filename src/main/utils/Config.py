# ______________

# ___,_______,_____

class Dataset:
    display = 1


class LSRound:
    needRandomAgent = 0
    needExplore = 0


class LSAgent:
    # "cude"
    deviceName = "cpu"
    # torch.device("cuda")
    # if torch.cuda.is_available() else torch.device("cpu")


class Instance:
    maxConstant = 1000
    instanceId = 1


class MCTS:
    selfplayMode = 0


def getShareResourcePath() -> str:
    import os
    import src.resource as resource
    shareResourcePath = os.path.split(resource.__file__)[0]
    for i in range(3):
        shareResourcePath = os.path.split(shareResourcePath)[0]
    shareResourcePath = os.path.join(shareResourcePath, "shareResource")
    assert os.path.exists(shareResourcePath)
    # print(type(shareResourcePath))
    # H:\JupyterNotebook\__\23year\7thMonth\shareResource
    # print(shareResourcePath)

    return shareResourcePath


def getResourcePath() -> str:
    import os
    import src.resource as resource
    resourcePath = os.path.split(resource.__file__)[0]

    assert os.path.exists(resourcePath)
    # print(type(shareResourcePath))
    # H:\JupyterNotebook\__\23year\7thMonth\shareResource
    # print(shareResourcePath)

    return resourcePath


if __name__ == '__main__':
    print(getResourcePath())
