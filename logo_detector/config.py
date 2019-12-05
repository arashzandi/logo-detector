import numpy as np

from logo_detector.detector import get_resnet

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    model = get_resnet()
    model_config = ([0.44943, 0.4331, 0.40244], [0.29053, 0.28417, 0.30194])