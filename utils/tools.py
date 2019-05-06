import random
import numpy as np
import argparse


class RandomState(object):
    def __init__(self):
        self.state = None
        self.npstate = None

    def GetState(self):
        self.state = random.getstate()
        self.npstate = np.random.get_state()

    def SetState(self):
        random.setstate(self.state)
        np.random.set_state(self.npstate)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        try:
            mod = getattr(mod, comp)
        except AttributeError:
            print(comp)
    return mod


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
