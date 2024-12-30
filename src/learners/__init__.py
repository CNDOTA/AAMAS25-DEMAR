from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .demix_learner import DEMLearner
from .sm2_learner import SM2Learner
from .td3_learner import TD3Learner
from .sub_avg_learner import SubAvgLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["demix_learner"] = DEMLearner
REGISTRY["sm2_learner"] = SM2Learner
REGISTRY["td3_learner"] = TD3Learner
REGISTRY["sub_avg_learner"] = SubAvgLearner
