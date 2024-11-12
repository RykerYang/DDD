def getShareResourcePath() -> str:
    import os
    import src.resource as resource
    shareResourcePath = os.path.split(resource.__file__)[0]
    for i in range(2):
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