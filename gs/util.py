"""
various utillty functions used outside the training process
"""
from gs.config import model_a


def model_to_ymal():
    print(model_a().to_yaml())


model_to_ymal()
