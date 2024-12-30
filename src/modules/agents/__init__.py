REGISTRY = {}

from .rnn_agent import RNNAgent
from .ernn_agent import ERNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ernn"] = ERNNAgent
