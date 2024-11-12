import multiprocessing
from typing import Dict

from flask import Flask, jsonify

from src.main.servlet_rl.g_agent import Agent


def create_shared_instance():
    manager = multiprocessing.Manager()
    return manager.proxy(Agent)


app = Flask(__name__)

# _________SharedClass__
shared_instance = create_shared_instance()


@app.route('/increment', methods=['POST'])
def increment():
    shared_instance.increment()
    return jsonify({'status': 'success'})


@app.route('/get_value', methods=['GET'])
def get_value():
    return jsonify({'value': shared_instance.get_value()})


if __name__ == '__main__':
    app.run(threaded=True)


class MultiThreadAgent(Agent):
    def __init__(self, modelManagerConfig: Dict):
        super().__init__(modelManagerConfig)
        # self.threadN = 1
        # self.threadList: List[Thread] = []
        # self.agentList: List[Agent] = []
        # self.agentList.append(self)
        # self.agentList.append
