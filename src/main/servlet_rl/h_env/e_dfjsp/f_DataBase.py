import os
import sqlite3
from typing import Dict, List, Tuple


class DataBase:
    def __init__(self, instanceId, shareResourcePath: str):
        self.dbPath = os.path.join(shareResourcePath, "sqlite3.db")
        self.instanceId = instanceId

    # Env
    def getFirstState(self, jobOpKey_remainNode: Dict):
        # __
        nodeEmbeddingList = []
        nodeEmbeddingIdList = []

        conn = sqlite3.connect(self.dbPath)
        cur = conn.cursor()
        # gm = GraphManager()
        for node in jobOpKey_remainNode.values():
            # nodeRs, nodeIdRs = gm.getEmbedding(node.jobId, node.opId)

            sql = """
                SELECT
                    node.embedding,
                    node.nodeId
                FROM
                    node
                WHERE
                    node.instanceId = {}
                    AND node.jobId = {}
                    AND node.opId = {}
                ORDER BY
                    node.nodeId ASC
                """.format(self.instanceId, node.jobId, node.opId)

            table = cur.execute(sql).fetchall()
            for row in table:
                nodeEmbeddingList.append(eval(row[0]))
                nodeEmbeddingIdList.append(row[1])

        return nodeEmbeddingList, nodeEmbeddingIdList

    def getState(self, jomList: List[Tuple[int, int, int]]):
        # __
        nodeEmbeddingList = []
        nodeEmbeddingIdList = []

        conn = sqlite3.connect(self.dbPath)
        cur = conn.cursor()
        # gm = GraphManager()
        for jom in jomList:
            # nodeRs, nodeIdRs = gm.getEmbedding(node.jobId, node.opId)

            sql = """
                SELECT
                    node.embedding,
                    node.nodeId 
                FROM
                    node 
                WHERE
                    node.instanceId = {}
                    AND node.jobId = {}
                    AND node.opId = {}
                    AND node.macId = {}
                ORDER BY
                    node.nodeId ASC
                """.format(self.instanceId, jom[0], jom[1], jom[2])

            table = cur.execute(sql).fetchall()
            nodeEmbeddingList.append(eval(table[0][0]))
            nodeEmbeddingIdList.append(table[0][1])

        return nodeEmbeddingList, nodeEmbeddingIdList

    def query_instance(self, start_codeId: int = -1) -> Tuple[List[List[List[int]]], List[int]]:
        # 0.__jobOpGraphId_start_codeId, __jobOpGraph_start_code:
        conn = sqlite3.connect(self.dbPath)
        cur = conn.cursor()
        sql1 = """
                SELECT
                    instance.instanceName,
                    instance.jobOpGraph 
                FROM
                    instance 
                WHERE
                    instance.id = {}
                """.format(self.instanceId)
        table_instance = cur.execute(sql1).fetchall()
        jobOpGraph: List[List[List[int]]] = eval(table_instance[0][1])

        if start_codeId < 0:
            return jobOpGraph, []
        sql2 = """
           SELECT
                start_code.startCode 
            FROM
                start_code 
            WHERE
                start_code.instanceId = {}
                AND start_code.id = {}

                """.format(self.instanceId, start_codeId)
        table_start_code = cur.execute(sql2).fetchall()
        start_code: List[int] = eval(table_start_code[0][0])
        conn.close()

        return jobOpGraph, start_code
