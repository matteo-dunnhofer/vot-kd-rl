"""
Written by Matteo Dunnhofer - 2020

Configuration class for A3CT, A3CTD
"""

class Configuration(object):


    DATA_PATH = ''
    CKPT_PATH = ''
    RESULTS_PATH = './trackers/results'
    REPORT_PATH = '../track/reports'

    SIZE = [128, 128]
    SEQ_LENGTH = 32
    CONTEXT_FACTOR = 1.5
    LSTM_UPDATE = False

    USE_RESULTS = True
    TRAST_TEACHER = 'SiamFC'

    USE_GPU = True

    def __init__(self):
        super(Configuration, self).__init__()

