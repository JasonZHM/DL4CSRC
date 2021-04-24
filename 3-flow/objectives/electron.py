import torch
import numpy as np
from torch.nn.functional import pdist
from .template import Target

class electron(Target):

    def __init__(self, numElectron):
        super(electron, self).__init__(2*numElectron,'electron') # num of variables, name
        self.numElectron = numElectron
        # self.wall = wall

    def energy(self, x):
        interaction = torch.sum(
            1/torch.stack(list(map(pdist, torch.split(x.reshape(-1, 2), self.numElectron))), dim=0).reshape(len(x), -1), 
            axis=1).reshape(-1, 1)
        wall = torch.sum((x**2), dim=1)/1000
        return interaction + wall