# -*- coding: utf-8 -*-

"""Main module."""

from .hl_cgp import HlCgp



class BinaryEdgeGt(object):
    def __init__(self, image):
        self.image = image




class HighLevelCgp(object):
    def __init__(self):
        pass


class Instance(object):
    def __init__(self, labels):
        
        # sanity check on input
        self.labels_sanity_check(labels)

        self.labels = labels

        # compute cgp
        self.hl_cgp = HlCgp(labels=labels)
    



    def labels_sanity_check(self, labels):
        assert labels.min() == 1

class TrainingInstane(Instance):
    def __init__(self, labels, gt):
        super(TrainingInstane, self).__init__(labels=labels)








if __name__ == "__main__":
    pass