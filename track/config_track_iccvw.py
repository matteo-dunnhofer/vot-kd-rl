"""
Written by Matteo Dunnhofer - 2020

Configuration class for A3CT, A3CTD
"""

class Configuration(object):


    DATA_PATH = '/media/TBData2/data/vot/'
    CKPT_PATH = '/media/TBData2/projects/vot-kd-rl-domain-adapt/experiments/VotRlDemo201910100651-pool/TEST_WORKER_25/ckpt/ActorCriticModel_120000.weights'  # None
    RESULTS_PATH = './trackers/results'

    SIZE = [128, 128]
    SEQ_LENGTH = 32
    CONTEXT_FACTOR = 1.5
    LSTM_UPDATE = False

    USE_RESULTS = True
    TRAST_TEACHER = 'SiamFC'

    USE_GPU = True

    def __init__(self):
        super(Configuration, self).__init__()

